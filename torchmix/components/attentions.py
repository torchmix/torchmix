import math
from typing import Tuple

from einops import einsum, rearrange, unpack
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class SelfAttention(Component):
    """A multi-head self attention layer.

    Examples:
        SelfAttention(dim=768, num_heads=8, head_dim=64)
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        self.inner_dim = num_heads * head_dim
        self.scale: float = head_dim**-0.5
        self.num_heads: float = num_heads

        self._to_qkv = nn.Linear(dim, 3 * self.inner_dim)
        self._proj = nn.Linear(self.inner_dim, dim)

    def to_qkv(
        self, x: Float[Tensor, "... n d"]
    ) -> Tuple[
        Float[Tensor, "... n d"],
        Float[Tensor, "... n d"],
        Float[Tensor, "... n d"],
    ]:
        query, key, value = unpack(
            self._to_qkv(x),
            [[self.inner_dim], [self.inner_dim], [self.inner_dim]],
            # einops.EinopsError: Invalid axis name ... in unpack(..., "... *")
            "b n *",
        )
        return query, key, value

    def split_qkv(
        self,
        query: Float[Tensor, "... n d_in"],
        key: Float[Tensor, "... n d_in"],
        value: Float[Tensor, "... n d_in"],
    ) -> Tuple[
        Float[Tensor, "... h n d_out"],
        Float[Tensor, "... h n d_out"],
        Float[Tensor, "... h n d_out"],
    ]:
        query, key, value = map(
            lambda x: rearrange(
                x,
                "... n (h d) -> ... h n d",
                h=self.num_heads,
            ),
            (query, key, value),
        )
        return query, key, value

    def attention(
        self,
        query: Float[Tensor, "... n d"],
        key: Float[Tensor, "... n d"],
        value: Float[Tensor, "... n d"],
    ) -> Float[Tensor, "... n d"]:
        dots = einsum(query, key, "... q d, ... k d -> ... q k") * self.scale
        attention = dots.softmax(dim=-1)
        out = einsum(attention, value, "... q k, ... k d -> ... q d")
        return out

    def collect_heads(
        self, out: Float[Tensor, "... h n d"]
    ) -> Float[Tensor, "... n h*d"]:
        return rearrange(out, "... h n d -> ... n (h d)")

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        q, k, v = self.to_qkv(x)
        q, k, v = self.split_qkv(q, k, v)
        out = self.attention(q, k, v)
        out = self.collect_heads(out)
        out = self._proj(out)
        return out


class WindowAttention(SelfAttention):
    """Local window attention layer from Swin-transformer

    Examples:
        WindowAttention(dim=96, window_size=8, num_heads=8, head_dim=64)
    """

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
        query: Float[Tensor, "... n d_in"],
        key: Float[Tensor, "... n d_in"],
        value: Float[Tensor, "... n d_in"],
    ) -> Tuple[
        Float[Tensor, "... h w head window d_out"],
        Float[Tensor, "... h w head window d_out"],
        Float[Tensor, "... h w head window d_out"],
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
