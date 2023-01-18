import math
from typing import List, Tuple

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .base import Attention
from .plugin import AttentionPlugin


class WindowAttention(Attention):
    """Local window attention layer from [Swin Transformer](https://arxiv.org/abs/2103.14030).

    Args:
        dim: The dimension size.
        num_heads: The number of attention heads.
        head_dim: The dimension size for each attention head.
        window_size: The window size for local attentions.
        plugins: A list of [`AttentionPlugin`](/plugins/AttentionPlugin)s to use.

    Examples:
        WindowAttention(dim=96, window_size=8, num_heads=8, head_dim=64)
    """

    def __init__(
        self,
        dim: int = 96,
        num_heads: int = 8,
        head_dim: int = 64,
        window_size: int = 8,
        plugins: List[AttentionPlugin] = [],
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            plugins=plugins,
        )
        self.window_size: int = window_size

    def split_heads(
        self,
        query: Float[Tensor, "... n d_in"],
        key: Float[Tensor, "... n d_in"],
        value: Float[Tensor, "... n d_in"],
    ) -> Tuple[
        Float[Tensor, "... h w head window d_out"],
        Float[Tensor, "... h w head window d_out"],
        Float[Tensor, "... h w head window d_out"],
    ]:
        _batch_size, _seq_len, _proj_dim = query.shape
        patch_size = round(math.sqrt(_seq_len))

        query, key, value = map(
            lambda x: rearrange(
                x,
                "... (h ph w pw) (hd d) -> ... h w hd (ph pw) d",
                ph=self.window_size,
                pw=self.window_size,
                h=patch_size // self.window_size,
                w=patch_size // self.window_size,
                hd=self.num_heads,
            ),
            (query, key, value),
        )
        return query, key, value

    def combine_heads(
        self,
        out: Float[Tensor, "... h w head window d_out"],
    ) -> Float[Tensor, "... n d_in"]:
        return rearrange(
            out,
            "... h w hd (ph pw) d -> ... (h ph w pw) (hd d)",
            ph=self.window_size,
            pw=self.window_size,
        )
