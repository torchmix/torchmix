import math

import torch
from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule
from torchmix.third_party.einops import EinMix, Rearrange, Reduce

__all__ = [
    "AvgPool",
    "Extract",
    "PatchEmbed",
    "PatchMerging",
    "PositionEmbed",
    "ChannelMixer",
    "TokenMixer",
    "Token",
]


class AvgPool(MixModule):
    def __init__(self):
        self.pool = Reduce("b n d -> b d", reduction="mean")

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
        return self.pool(x)


class PatchEmbed(MixModule):
    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        dim: int = 768,
    ):
        self.projection = EinMix(
            "b c (h ph) (w pw) -> b (h w) d",
            weight_shape="c ph pw d",
            bias_shape="d",
            ph=patch_size,
            pw=patch_size,
            c=channels,
            d=dim,
        )

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b n d"]:
        return self.projection(x)


class ChannelMixer(MixModule):
    def __init__(
        self,
        act_layer: Partial[MixModule],
        dim: int = 1024,
        expansion_factor: float = 4,
    ):
        self.block = nn.Sequential(
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=dim,
                d_out=int(dim * expansion_factor),
            ),
            act_layer(),
            EinMix(
                "b n d_out -> b n d_in",
                weight_shape="d_out d_in",
                bias_shape="d_in",
                d_in=dim,
                d_out=int(dim * expansion_factor),
            ),
        )

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        return self.block(x)


class TokenMixer(MixModule):
    def __init__(
        self,
        act_layer: Partial[MixModule],
        seq_length: int = 196,
        expansion_factor: float = 0.5,
    ):
        self.block = nn.Sequential(
            EinMix(
                "b n_in d -> b n_out d",
                weight_shape="n_in n_out",
                bias_shape="n_out",
                n_in=seq_length,
                n_out=int(seq_length * expansion_factor),
            ),
            act_layer(),
            EinMix(
                "b n_out d -> b n_in d",
                weight_shape="n_out n_in",
                bias_shape="n_in",
                n_in=seq_length,
                n_out=int(seq_length * expansion_factor),
            ),
        )

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        return self.block(x)


class Token(MixModule):
    def __init__(
        self,
        dim: int = 1024,
    ):
        self.class_token = nn.Parameter(torch.zeros(dim))

    def forward(self, *_) -> Float[Tensor, " d"]:
        return self.class_token


class PositionEmbed(MixModule):
    def __init__(
        self,
        seq_length: int = 197,
        dim: int = 1024,
    ):
        self.pos_embed = nn.Parameter(torch.randn(seq_length, dim)) * 0.02

    def forward(self, *_) -> Float[Tensor, "n d"]:
        return self.pos_embed


class Extract(MixModule):
    def __init__(self, index: int):
        self.index = index

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
        return x[:, self.index, :]


class PatchMerging(MixModule):
    def __init__(self, dim: int = 96):
        self.merge = Rearrange.partial(
            "b (h ph w pw) d -> b (h w) (ph pw d)",
            ph=2,
            pw=2,
            h=None,
            w=None,
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 4),
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=dim * 4,
                d_out=dim * 2,
            ),
        )

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n/4 d*2"]:
        _batch_size, _seq_length, _dim = x.shape

        x = self.merge(
            h=round(math.sqrt(_seq_length / 4)),
            w=round(math.sqrt(_seq_length / 4)),
        )(x)

        return self.proj(x)
