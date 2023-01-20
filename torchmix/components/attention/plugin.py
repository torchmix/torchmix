# TODO: Implement caching mechanisms for non-trainable plugins
# TODO: Implement dynamic length handling
# TODO: Implement mode argument between ViT <-> LM
# TODO: Maybe class level global cache?

import math

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core.component import Component


class AttentionPlugin(Component):
    """Base class for all plugins for Attention.

    Examples:
        class CustomPlugin(AttentionPlugin): ...
    """

    def pre_qkv(self, x):
        return x

    def post_qkv(self, q, k, v):
        return q, k, v

    def pre_split_heads(self, q, k, v):
        return q, k, v

    def post_split_heads(self, q, k, v):
        return q, k, v

    def pre_scaled_dot(self, q, k):
        return q, k

    def post_scaled_dot(self, dots):
        return dots

    def pre_softmax(self, dots):
        return dots

    def post_softmax(self, attention):
        return attention

    def pre_weighted_value(self, attention):
        return attention

    def post_weighted_value(self, out):
        return out

    def pre_combine_heads(self, out):
        return out

    def post_combine_heads(self, out):
        return out

    def pre_projection(self, out):
        return out

    def post_projection(self, out):
        return out


class CausalMask(AttentionPlugin):
    """Mask query-key dot products for causal attention.

    Examples:
        Attention(
            dim=768,
            plugins=[
                CausalMask()
            ],
        )
    """

    def __init__(self):
        self._seq_len_cached = None
        self._mask_cached = None

    def _update_cache(
        self, dots: Float[Tensor, "... q k"], seq_dimension: int = -1
    ):
        seq_len = dots.shape[seq_dimension]

        if (
            not self._seq_len_cached
            or seq_len > self._seq_len_cached
            or dots.device != self._mask_cached.device
        ):
            self._seq_len_cached = seq_len

            self._mask_cached = ~torch.tril(
                torch.ones(
                    seq_len,
                    seq_len,
                    dtype=torch.bool,
                    device=dots.device,
                ),
            )

        return seq_len

    def pre_softmax(self, dots: Tensor):
        seq_len = self._update_cache(dots)

        return dots.masked_fill(
            self._mask_cached[:seq_len, :seq_len],
            -torch.finfo(dots.dtype).max,
        )


class DropProjection(AttentionPlugin):
    """Apply dropout after projection layer.

    Examples:
        Attention(
            dim=768,
            plugins=[
                DropProjection(p=0.1)
            ],
        )
    """

    def __init__(self, p: float = 0.1):
        self.dropout = nn.Dropout(p)

    def post_projection(self, out):
        return self.dropout(out)


class DropAttention(AttentionPlugin):
    """Apply dropout for attentions.

    Examples:
        Attention(
            dim=768,
            plugins=[
                DropAttention(p=0.1)
            ],
        )
    """

    def __init__(self, p: float = 0.1):
        self.dropout = nn.Dropout(p)

    def post_softmax(self, attention):
        return self.dropout(attention)


class SubNorm(AttentionPlugin):
    """Apply layer normalization before projection.

    This plugin implements Sub-LN for [Foundation Transformers](https://arxiv.org/pdf/2210.06423.pdf).
    Note that Sub-LN presumes Pre-LN rather than Post-LN

    Examples:
        PreNorm(
            Attention(
                dim=768,
                plugins=[
                    SubNorm(dim=768)
                ],
            ),
            dim=768,
        )
    """

    def __init__(self, dim: int = 768):
        self.norm = nn.LayerNorm(dim)

    def pre_projection(self, out):
        return self.norm(out)


class RelativePositionBiasViT(AttentionPlugin):
    """Relative Position Bias for [Swin Transformer](https://arxiv.org/abs/2103.14030).

    The relative position bias is intended to be added to the attention before
    the softmax is applied. It helps to improve the attention mechanism by
    incorporating the relative positions of the elements in the input.

    Note that the number of parameters is `O(window_size)`, not `O(window_size**2)`.

    Args:
        window_size: The window size for which attention is computed.
        num_heads: Number of heads for the multi-head attention mechanism.

    Examples:
        WindowAttention(
            dim=768,
            num_heads=12,
            plugins=[
                RelativePositionBiasViT(
                    window_size=8,
                    num_heads=12,
                )
            ],
        )
    """

    relative_position_index: Tensor

    def __init__(
        self,
        window_size: int = 8,
        num_heads: int = 8,
    ):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords_flatten = rearrange(
            torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")),
            "xy h w -> xy (h w)",
        )
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = rearrange(relative_coords, "xy h w -> h w xy")
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                num_heads, (2 * window_size - 1) * (2 * window_size - 1)
            )
        )

        nn.init.trunc_normal_(self.relative_position_bias_table)

    def post_scaled_dot(
        self, dots: Float[Tensor, "... h q k"]
    ) -> Float[Tensor, "... h q k"]:
        return (
            dots
            + self.relative_position_bias_table[:, self.relative_position_index]
        )


class RelativePositionBias(AttentionPlugin):
    """Relative Position Bias for [T5](https://arxiv.org/abs/1910.10683).

    Examples:
        Attention(
            dim=768,
            num_heads=12,
            plugins=[
                RelativePositionBias(
                    num_buckets=256,
                    num_heads=12,
                )
            ],
        )
    """

    def __init__(
        self,
        num_buckets: int = 32,
        num_heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.relative_attention_bias = nn.Parameter(
            torch.randn(num_heads, num_buckets)
        )

        self._seq_len_cached = None
        self._relative_position_bucket_cached = None

    def _update_relative_position_bucket(self):
        seq_len = self._seq_len_cached
        index = torch.arange(seq_len, dtype=torch.long)
        relative_position = index[None, :] - index[:, None]

        ret = 0
        n = -relative_position
        if not self.causal:
            self.num_buckets //= 2
            ret += (n < 0).long() * self.num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = self.num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(seq_len / max_exact)
                * (self.num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, self.num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)

        self._relative_position_bucket_cached = ret

    def _update_cache(
        self, dots: Float[Tensor, "... q k"], seq_dimension: int = -1
    ):
        seq_len = dots.shape[seq_dimension]

        if not self._seq_len_cached or seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            self._update_relative_position_bucket()

        return seq_len

    def post_scaled_dot(
        self, dots: Float[Tensor, "... h q k"]
    ) -> Float[Tensor, "... h q k"]:
        self._update_cache(dots)

        bias = self.relative_attention_bias[
            :, self._relative_position_bucket_cached
        ]

        print(dots.shape, bias.shape)
        return dots + bias


class RotaryEmbedding(AttentionPlugin):
    """Rotary Position Embedding for [RoFormer](https://arxiv.org/abs/1910.10683).

    Args:
        head_dim: The dimension size for each attention head.
        seq_len: The length of given sequence.

    Examples:
        Attention(
            dim=768,
            num_heads=12,
            head_dim= 64,
            plugins=[
                RotaryEmbedding(
                    head_dim=64,
                )
            ],
        )
    """

    inv_freq: Tensor

    def __init__(
        self,
        head_dim: int = 64,
    ):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    @staticmethod
    def rotate(x):
        x1, x2 = rearrange(x, "... (half d) -> ... half d", half=2).unbind(-2)
        return rearrange([-x2, x1], "half ... d -> ... (half d)")

    def apply_rotary_embedding(self, x: Tensor) -> Tensor:
        return x * self._cos_cached + self.rotate(x) * self._sin_cached

    def _update_cache(
        self, x: Float[Tensor, "... n d"], seq_dimension: int = -2
    ):
        seq_len = x.shape[seq_dimension]

        if (
            seq_len != self._seq_len_cached
            or self._cos_cached.device != x.device
            or self._cos_cached.dtype != x.dtype
        ):
            self._seq_len_cached = seq_len

            index = torch.arange(seq_len)
            freqs = einsum(index, self.inv_freq, "i, j -> i j")
            emb = repeat(freqs, "... theta -> ... (n theta)", n=2)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

    def post_split_heads(self, q, k, v):
        self._update_cache(q)
        return (
            self.apply_rotary_embedding(q),
            self.apply_rotary_embedding(k),
            v,
        )
