import math
from typing import Literal, Optional

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


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
                CausalMask(
                    block_size=1024,
                    mode="static",
                )
            ],
        )
    """

    def __init__(
        self,
        block_size: Optional[int] = None,
        mode: Literal["static", "dynamic"] = "dynamic",
    ):
        self.mode = mode

        if mode not in ("static", "dynamic"):
            raise TypeError("mode must be one of 'dynamic' or 'static'")

        if mode == "dynamic" and block_size:
            raise TypeError("block_size must be None for dynamic mode.")

        if mode == "static":
            self.register_buffer(
                "mask",
                ~torch.tril(
                    torch.ones(block_size, block_size, dtype=torch.bool),
                ),
            )

    def pre_softmax(self, dots):
        _seq_length = dots.shape[-1]

        if self.mode == "static":
            return dots.masked_fill(
                self.mask[:_seq_length, :_seq_length], float("-inf")
            )

        return dots.masked_fill(
            ~torch.tril(
                torch.ones(_seq_length, _seq_length, dtype=torch.bool),
            ),
            float("-inf"),
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


class SubLayerNorm(AttentionPlugin):
    """Apply layer normalization before projection.

    This plugin implements Sub-LN for [Foundation Transformers](https://arxiv.org/pdf/2210.06423.pdf).
    Note that Sub-LN presumes Pre-LN rather than Post-LN

    Examples:
        PreNorm(
            Attention(
                dim=768,
                plugins=[
                    SubLayerNorm(dim=768)
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
                    seq_length=1024,
                    num_buckets=256,
                    num_heads=12,
                )
            ],
        )
    """

    def __init__(
        self,
        seq_length: int = 128,
        num_buckets: int = 32,
        num_heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.num_buckets = num_buckets
        self.seq_length = seq_length
        self.relative_attention_bias = nn.Parameter(
            torch.randn(num_heads, num_buckets)
        )

    def _relative_position_bucket(
        self,
        relative_position,
    ):
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
                / math.log(self.seq_length / max_exact)
                * (self.num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, self.num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)

        return ret

    def post_scaled_dot(
        self, dots: Float[Tensor, "... h q k"]
    ) -> Float[Tensor, "... h q k"]:
        q, k = dots.shape[-2:]
        q_pos = torch.arange(k - q, k, dtype=torch.long, device=dots.device)
        k_pos = torch.arange(k, dtype=torch.long, device=dots.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos)
        bias = self.relative_attention_bias[:, rp_bucket]
        return dots + bias
