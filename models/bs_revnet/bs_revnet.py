'''
Credit:
https://github.com/karttikeya/minREV Ze Liu, Yutong Lin, Yixuan Wei
'''

from functools import partial

import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from models.bs_roformer.attend import Attend

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from torch.autograd import Function as Function
import sys

# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


# norm

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flushing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.

    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(ctx, x, layers):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """

        X_1, X_2 = torch.chunk(x, 2, dim=-1) # split the concat X

        for _, lyr in enumerate(layers):
            X_1, X_2 = lyr(X_1, X_2) # lyr returns Y1, Y2 (inplace to reduce footprint)
            all_tensors = [X_1.detach(), X_2.detach()] # save for backward pass
        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1) # concat X

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass. Any intermediate activations from `buffer_layers` are
        recovered from ctx. Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2 = ctx.saved_tensors
        layers = ctx.layers

        for lyr in layers[::-1]:
            X_1, X_2, dX_1, dX_2 = lyr.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2
            )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class RevBackPropFast(RevBackProp):
    @staticmethod
    def backward(ctx, dx):
        """Overwrite backward by using PyTorch Streams to parallelize."""

        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2, *int_tensors = ctx.saved_tensors

        layers = ctx.layers

        # Keep a dictionary of events to synchronize on
        # Each is for the completion of a recalculation (f) or gradient calculation (b)
        events = {}
        for i in range(len(layers)):
            events[f"f{i}"] = torch.cuda.Event()
            events[f"b{i}"] = torch.cuda.Event()

        # Run backward staggered on two streams, which were defined globally in __init__
        # Initial pass
        with torch.cuda.stream(s1):
            layer = layers[-1]
            prev = layer.backward_pass_recover(
                Y_1=X_1, Y_2=X_2
            )

            events["f0"].record(s1)

        # Stagger streams based on iteration
        for i, (this_layer, next_layer) in enumerate(
            zip(layers[1:][::-1], layers[:-1][::-1])
        ):
            if i % 2 == 0:
                stream1 = s1
                stream2 = s2
            else:
                stream1 = s2
                stream2 = s1

            with torch.cuda.stream(stream1):
                # b{i} waits on b{i-1}
                if i > 0:
                    events[f"b{i-1}"].synchronize()

                if i % 2 == 0:
                    dY_1, dY_2 = this_layer.backward_pass_grads(
                        *prev, dX_1, dX_2
                    )
                else:
                    dX_1, dX_2 = this_layer.backward_pass_grads(
                        *prev, dY_1, dY_2
                    )

                events[f"b{i}"].record(stream1)

            with torch.cuda.stream(stream2):
                # f{i} waits on f{i-1}
                events[f"f{i}"].synchronize()

                prev = next_layer.backward_pass_recover(
                    Y_1=prev[0], Y_2=prev[1]
                )

                events[f"f{i+1}"].record(stream2)

        # Last iteration
        if len(layers) - 1 % 2 == 0:
            stream2 = s1
        else:
            stream2 = s2
        next_layer = layers[0]

        with torch.cuda.stream(stream2):
            # stream2.wait_event(events[f"b{len(layers)-2}_end"])
            if len(layers) > 1:
                events[f"b{len(layers)-2}"].synchronize()
            if len(layers) - 1 % 2 == 0:
                dY_1, dY_2 = next_layer.backward_pass_grads(*prev, dX_1, dX_2)
                dx = torch.cat([dY_1, dY_2], dim=-1)
            else:
                dX_1, dX_2 = next_layer.backward_pass_grads(*prev, dY_1, dY_2)
                dx = torch.cat([dX_1, dX_2], dim=-1)
            events[f"b{len(layers)-1}"].record(stream2)

        # Synchronize, for PyTorch 1.9
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        events[f"b{len(layers)-1}"].synchronize()

        del int_tensors
        del dX_1, dX_2, dY_1, dY_2, X_1, X_2, prev[:]
        return dx, None


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            x = self.norm(x)
    
            q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
    
            if exists(self.rotary_embed):
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)
    
            out = self.attend(q, k, v)
    
            gates = self.to_gates(x)
            out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()
    
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)


class DropPath(Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    Ref https://github.com/facebookresearch/dinov2/blob/main/dinov2/layers/drop_path.py#L26
    """
    
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if not self.training:
            return x

        with torch.cuda.amp.autocast(enabled=True):
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
            if keep_prob > 0.0 and self.scale_by_keep:
                random_tensor.div_(keep_prob)
    
            return x * random_tensor


class ReversibleLayer(Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            rotary_embed=None,
            flash_attn=True,
            drop_prob=0.
    ):
        super().__init__()

        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, rotary_embed=rotary_embed, flash=flash_attn)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.drop_path = DropPath(drop_prob=drop_prob) if drop_prob else nn.Identity()
        self.seeds = {}
    
    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.

        From RevViT.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X1, X2):
        """Reversible forward function.
        Y_1 = X_1 + Attn(X_2)
        Y_2 = X_2 + FF(Y_1)
        """

        self.seed_cuda("attn")
        attn_out = self.attn(X2)
        self.seed_cuda("droppath")
        Y1 = X1 + self.drop_path(attn_out)
        del X1

        self.seed_cuda("ff")
        ff_out = self.ff(Y1)
        torch.manual_seed(self.seeds["droppath"])
        Y2 = X2 + self.drop_path(ff_out)
        del X2

        return Y1, Y2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2):
        """
        equations for recovering activations:
        X2 = Y2 - FF(Y1)
        X1 = Y1 - Attn(X2)
        """

        # gradient calculcation of FF
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["ff"])
            g_Y_1 = self.ff(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation
        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # gradient calculcation of attn
        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.attn(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

            f_X_2.backward(dY_1, retain_graph=True)

        # activation recomputation
        with torch.no_grad():
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2

    def backward_pass_recover(self, Y_1, Y_2):
        """
        Use equations to recover activations and return them.
        Used for streaming the backward pass.
        """

        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["ff"])
            g_Y_1 = self.ff(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.attn(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        # Keep tensors around to do backprop on the graph.
        ctx = [X_1, X_2, Y_1, g_Y_1, f_X_2]
        return ctx

    def backward_pass_grads(self, X_1, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        """
        Receive intermediate activations and inputs to backprop through.
        """

        with torch.enable_grad():
            g_Y_1.backward(dY_2)

        with torch.no_grad():
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        with torch.enable_grad():
            f_X_2.backward(dY_1)

        with torch.no_grad():
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None
            X_2.detach()

        return dY_1, dY_2

    def backward_default(self, x, layers)
        X_1, X_2 = torch.chunk(x, 2, dim=-1) # split the concat X

        for _, lyr in enumerate(layers):
            X_1, X_2 = lyr(X_1, X_2) # lyr returns Y1, Y2 (inplace to reduce footprint)

        return torch.cat([X_1, X_2], dim=-1) # concat X
    
class ReversibleTransformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            fast_backprop=False,
            drop_prob=0.1
    ):
        super().__init__()

        if not self.training:
            self.executing_fn = ReversibleLayer.backward_default
        elif fast_backprop:
            global s1, s2
            s1 = torch.cuda.default_stream(device=torch.cuda.current_device())
            s2 = torch.cuda.Stream(device=torch.cuda.current_device())
            self.executing_fn = RevBackPropFast.apply
        else:
            self.executing_fn = RevBackProp.apply

        self.layers = ModuleList([])

        for _ in range(depth):
                self.layers.append(ReversibleLayer(
                        dim=dim,
                        dim_head=dim_head,
                        heads=heads,
                        attn_dropout=attn_dropout,
                        ff_dropout=ff_dropout,
                        ff_mult=ff_mult,
                        rotary_embed=rotary_embed,
                        flash_attn=flash_attn,
                        drop_prob=drop_prob
                    )
                )
            
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        x = self.executing_fn(x, self.layers)
        x = self.norm(x)

        return x 


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2) 

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)

class TwoStreamFusion(Module):
    def __init__(
            self, 
            dim,
            hidden_dim=None,
            out_dim=None,
            act_layer=nn.GELU,
            dropout=0.
            ):
        """
        Module for fusing both streams of the reversible model by concatenation,
        then applying an MLP with a hidden dim of dim*2 and output dim of dim
        to downsample.

        https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
        """
        super().__init__()

        out_dim = out_dim or int(dim / 2)
        hidden_dim = hidden_dim or dim
        self.fuse_fn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                act_layer(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.fuse_fn(x)

# main class

DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRevnet(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            fast_backprop=False, # speed improvements seen @ time_transformer_depth or freq_transformer_depth > 1
            drop_prob=0.2,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            fast_backprop=fast_backprop,
            drop_prob=drop_prob,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            transformer_modules = nn.ModuleList([
                ReversibleTransformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, **transformer_kwargs),
                ReversibleTransformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, **transformer_kwargs)
            ])
            self.layers.append(transformer_modules)

        self.fuse = TwoStreamFusion(dim * 2) # b f t d*2 -> b t f d
        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        with torch.cuda.amp.autocast(enabled=True):
        
            device = raw_audio.device
    
            if raw_audio.ndim == 2:
                raw_audio = rearrange(raw_audio, 'b t -> b 1 t')
    
            channels = raw_audio.shape[1]
            assert (not self.stereo and channels == 1) or (
                        self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'
    
            # to stft
    
            raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')
    
            stft_window = self.stft_window_fn(device=device)
    
            stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
            stft_repr = torch.view_as_real(stft_repr)
    
            stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
            stft_repr = rearrange(stft_repr,
                                  'b s f t c -> b (f s) t c')  # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
    
            x = rearrange(stft_repr, 'b f t c -> b t (f c)')
    
            x = self.band_split(x)

            # axial / hierarchical attention
            
            x = torch.cat([x, x], dim=-1) # b t f d -> b f t d*2

        for time_transformer, freq_transformer in self.layers:
            
            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')
            
            x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x = freq_transformer(x)
            x, = unpack(x, ps, '* f d')

        with torch.cuda.amp.autocast(enabled=True):
            
            x = self.fuse(x) # b f t d*2 -> b t f d
    
            x = self.final_norm(x) 
    
            num_stems = len(self.mask_estimators)
    
            mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
            mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)
    
            # modulate frequency representation
    
            stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')
    
            # complex number multiplication
    
            stft_repr = torch.view_as_complex(stft_repr)
            mask = torch.view_as_complex(mask)
    
            stft_repr = stft_repr * mask
    
            # istft
    
            stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
    
            recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False)
    
            recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)
    
            if num_stems == 1:
                recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')
    
            # if a target is passed in, calculate loss for learning
    
            if not exists(target):
                return recon_audio
    
            if self.num_stems > 1:
                assert target.ndim == 4 and target.shape[1] == self.num_stems
    
            if target.ndim == 2:
                target = rearrange(target, '... t -> ... 1 t')
    
            target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft
    
            loss = F.l1_loss(recon_audio, target)
    
            multi_stft_resolution_loss = 0.
    
            for window_size in self.multi_stft_resolutions_window_sizes:
                res_stft_kwargs = dict(
                    n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                    win_length=window_size,
                    return_complex=True,
                    window=self.multi_stft_window_fn(window_size, device=device),
                    **self.multi_stft_kwargs,
                )
    
                recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)
    
                multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)
    
            weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
    
            total_loss = loss + weighted_multi_resolution_loss
    
            if not return_loss_breakdown:
                return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
