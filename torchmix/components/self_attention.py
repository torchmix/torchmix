from typing import Tuple

from einops import einsum, rearrange, unpack
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


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

        # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
        # self._to_qkv = EinMix(
        #     "... d_in -> ... d_out",
        #     weight_shape="d_in d_out",
        #     bias_shape="d_out",
        #     d_in=dim,
        #     d_out=3 * self.inner_dim,
        # )
        self._to_qkv = nn.Linear(dim, 3 * self.inner_dim)

        # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
        # self._proj = EinMix(
        #     "... d_out -> ... d_in",
        #     weight_shape="d_in d_out",
        #     bias_shape="d_in",
        #     d_in=dim,
        #     d_out=self.inner_dim,
        # )
        self._proj = nn.Linear(self.inner_dim, dim)

    def to_qkv(
        self, x: Float[Tensor, "... d"]
    ) -> Tuple[Float[Tensor, "... d"], Float[Tensor, "... d"], Float[Tensor, "... d"],]:
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
