import math
from typing import Tuple

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from .self_attention import SelfAttention


class WindowAttention(SelfAttention):
    def __init__(
        self,
        dim: int = 96,
        window_size: int = 8,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.window_size: int = window_size

    def split_qkv(
        self,
        query: Float[Tensor, "b n d_in"],
        key: Float[Tensor, "b n d_in"],
        value: Float[Tensor, "b n d_in"],
    ) -> Tuple[
        Float[Tensor, "b h w head window d_out"],
        Float[Tensor, "b h w head window d_out"],
        Float[Tensor, "b h w head window d_out"],
    ]:
        _batch_size, _seq_length, _inner_dim = query.shape
        patch_size = round(math.sqrt(_seq_length))

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

    def collect_heads(
        self,
        out: Float[Tensor, "... h w head window d_out"],
    ) -> Float[Tensor, "... n d_in"]:
        return rearrange(
            out,
            "... h w hd (ph pw) d -> ... (h ph w pw) (hd d)",
            ph=self.window_size,
            pw=self.window_size,
        )
