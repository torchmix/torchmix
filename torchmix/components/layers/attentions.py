import math

from einops import einsum, rearrange, unpack
from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule
from torchmix.third_party.einops import EinMix

__all__ = [
    "SelfAttention",
    "WindowAttention",
]


class SelfAttention(MixModule):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
    ):
        self.inner_dim = num_heads * head_dim
        self.scale: float = head_dim**-0.5
        self.num_heads: float = num_heads

        self._to_qkv = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            bias_shape="d_out",
            d_in=dim,
            d_out=3 * self.inner_dim,
        )

        self._proj = EinMix(
            "b n d_out -> b n d_in",
            weight_shape="d_in d_out",
            bias_shape="d_in",
            d_in=dim,
            d_out=self.inner_dim,
        )

    def to_qkv(
        self, x: Float[Tensor, "b n d"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b n d"], Float[Tensor, "b n d"]]:
        query, key, value = unpack(
            self._to_qkv(x),
            [[self.inner_dim], [self.inner_dim], [self.inner_dim]],
            "b n *",
        )
        return query, key, value

    def split_qkv(
        self,
        query: Float[Tensor, "b n d_in"],
        key: Float[Tensor, "b n d_in"],
        value: Float[Tensor, "b n d_in"],
    ) -> tuple[
        Float[Tensor, "b h n d_out"],
        Float[Tensor, "b h n d_out"],
        Float[Tensor, "b h n d_out"],
    ]:
        query, key, value = map(
            lambda x: rearrange(
                x,
                "b n (h d) -> b h n d",
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

    def collect_heads(self, out: Float[Tensor, "b h n d"]) -> Float[Tensor, "b n h*d"]:
        return rearrange(out, "b h n d -> b n (h d)")

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        q, k, v = self.to_qkv(x)
        q, k, v = self.split_qkv(q, k, v)
        out = self.attention(q, k, v)
        out = self.collect_heads(out)
        out = self._proj(out)
        return out


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
        query: Float[Tensor, "b n d"],
        key: Float[Tensor, "b n d"],
        value: Float[Tensor, "b n d"],
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b n d"], Float[Tensor, "b n d"],]:
        _batch_size, _seq_length, _inner_dim = query.shape
        patch_size = round(math.sqrt(_seq_length))

        query, key, value = map(
            lambda x: rearrange(
                x,
                "b (h ph w pw) (hd d) -> b h w hd (ph pw) d",
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
        self, out: Float[Tensor, "b h n d_out"]
    ) -> Float[Tensor, "b n d_in"]:
        return rearrange(
            out,
            "b h w hd (ph pw) d -> b (h ph w pw) (hd d)",
            ph=self.window_size,
            pw=self.window_size,
        )
