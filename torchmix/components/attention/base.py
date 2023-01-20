from typing import List, Optional, Tuple

import torch
from einops import einsum, rearrange, unpack
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core.component import Component

from .plugin import AttentionPlugin


class Attention(Component):
    """Base class for all multi-head self attentions.

    Args:
        dim: The dimension size.
        num_heads: The number of attention heads.
        head_dim: The dimension size for each attention head.
        plugins: A list of [`AttentionPlugin`](/plugins/AttentionPlugin)s to use.

    Examples:
        Attention(dim=768, num_heads=8, head_dim=64, plugins=[])
    """

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        plugins: List[AttentionPlugin] = [],
    ):
        head_dim = head_dim or (dim // num_heads)

        self.proj_dim = num_heads * head_dim
        self.scale: float = head_dim**-0.5
        self.num_heads = num_heads

        self._qkv = nn.Linear(dim, 3 * self.proj_dim)
        self._proj = nn.Linear(self.proj_dim, dim)

        self.plugins: List[AttentionPlugin] = nn.ModuleList(plugins)

    def qkv(
        self, x: Float[Tensor, "... n d"]
    ) -> Tuple[
        Float[Tensor, "... n d"],
        Float[Tensor, "... n d"],
        Float[Tensor, "... n d"],
    ]:
        query, key, value = unpack(
            self._qkv(x),
            [[self.proj_dim], [self.proj_dim], [self.proj_dim]],
            "b n *",
        )
        return query, key, value

    def split_heads(
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

    def combine_heads(
        self, out: Float[Tensor, "... h n d"]
    ) -> Float[Tensor, "... n h*d"]:
        return rearrange(out, "... h n d -> ... n (h d)")

    def scaled_dot(
        self, query: Float[Tensor, "... q d"], key: Float[Tensor, "... k d"]
    ) -> Float[Tensor, "... q k"]:
        return einsum(query, key, "... q d, ... k d -> ... q k") * self.scale

    def softmax(
        self, dots: Float[Tensor, "... q k"]
    ) -> Float[Tensor, "... q k"]:
        return torch.softmax(dots, dim=-1)

    def weighted_value(
        self,
        attention: Float[Tensor, "... q k"],
        value: Float[Tensor, "... k d"],
    ) -> Float[Tensor, "... q d"]:
        return einsum(attention, value, "... q k, ... k d -> ... q d")

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... n d"]:
        for plugin in self.plugins:
            x = plugin.pre_qkv(x)
        query, key, value = self.qkv(x)
        for plugin in self.plugins:
            query, key, value = plugin.post_qkv(query, key, value)

        for plugin in self.plugins:
            query, key, value = plugin.pre_split_heads(query, key, value)
        query, key, value = self.split_heads(query, key, value)
        for plugin in self.plugins:
            query, key, value = plugin.post_split_heads(query, key, value)

        for plugin in self.plugins:
            query, key = plugin.pre_scaled_dot(query, key)
        dots = self.scaled_dot(query, key)
        for plugin in self.plugins:
            dots = plugin.post_scaled_dot(dots)

        for plugin in self.plugins:
            dots = plugin.pre_softmax(dots)
        attention = self.softmax(dots)
        for plugin in self.plugins:
            attention = plugin.post_softmax(attention)

        for plugin in self.plugins:
            attention = plugin.pre_weighted_value(attention)
        out = self.weighted_value(attention, value)
        for plugin in self.plugins:
            out = plugin.post_weighted_value(out)

        for plugin in self.plugins:
            out = plugin.pre_combine_heads(out)
        out = self.combine_heads(out)
        for plugin in self.plugins:
            out = plugin.post_combine_heads(out)

        for plugin in self.plugins:
            out = plugin.pre_projection(out)
        out = self._proj(out)
        for plugin in self.plugins:
            out = plugin.post_projection(out)

        return out
