import math
from typing import Optional

import torch
from einops import repeat
from jaxtyping import Float, Integer
from torch import Tensor

from torchmix import nn
from torchmix.core.component import Component
from torchmix.third_party.einops import Rearrange


class PositionalEmbedding(Component):
    """Learnable Positional embeddings

    Examples:
        PositionalEmbedding(seq_len=196, dim=768, gain=1.0)
    """

    def __init__(
        self,
        dim: int = 768,
        seq_len: int = 768,
        gain: float = 1.0,
    ):
        self.embedding = nn.Parameter(torch.randn(seq_len, dim)) * 0.02 * gain

    def forward(self, *_, **__) -> Float[Tensor, "n d"]:
        return self.embedding


class SinusoidalEmbedding(Component):
    """Sinusoidal Positional embeddings in [Attention is all you need](https://arxiv.org/abs/1706.03762)

    Examples:
        SinusoidalEmbedding(dim=768)
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

        self._seq_len_cached = None
        self._embedding_cached = None

    def _update_cache(self, x: Tensor, seq_dimension: int = -2):
        seq_len = x.shape[seq_dimension]

        if not self._seq_len_cached or seq_len > self._seq_len_cached:
            print("Cache..")
            self._seq_len_cached = seq_len

            position = repeat(
                torch.arange(seq_len, device=x.device, dtype=x.dtype),
                "n -> n d",
                d=self.dim,
            )

            dim = torch.arange(self.dim, device=x.device, dtype=x.dtype)
            dim = torch.exp(-math.log(10000) * (2 * (dim // 2) / self.dim))
            dim = repeat(
                dim,
                "d -> n d",
                n=seq_len,
            )

            position = position * dim

            position[:, 0::2] = position[:, 0::2].sin()
            position[:, 1::2] = position[:, 1::2].cos()

            self._embedding_cached = position

        return seq_len

    def forward(
        self, x: Float[Tensor, "... n d"], *_, **__
    ) -> Float[Tensor, "n d"]:
        seq_len = self._update_cache(x)

        return self._embedding_cached[
            :seq_len,
        ]


class VocabEmbedding(Component):
    def __init__(
        self,
        dim: int = 768,
        vocab_size: int = 50257,
        gain: float = 1.0,
        embedding: Optional[nn.Embedding] = None,
    ):
        self.embedding = embedding or nn.Embedding(vocab_size, dim)

        torch.nn.init(self.embedding.weight, std=0.02 * gain)

    def forward(self, x: Integer[Tensor, "... n"]) -> Float[Tensor, "... n d"]:
        return self.embedding(x)


class PatchEmbedding(Component):
    """A layer that convert image into patch embeddings.

    Examples:
        PatchEmbedding(patch_size=16, channels=3, dim=768)
    """

    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        dim: int = 768,
    ):
        # Could be merged into single EinMix Layer later
        # EinMix("... c (h ph) (w pw) -> ... (h w) d")
        self.proj = nn.Sequential(
            Rearrange(
                "... c (h ph) (w pw) -> ... (h w) (c ph pw)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(patch_size * patch_size * channels, dim),
        )

    def forward(
        self, x: Float[Tensor, "... c h w"]
    ) -> Float[Tensor, "... n d"]:
        return self.proj(x)


class ClassEmbedding(Component):
    """A single token that represents CLS

    Examples:
        ClassEmbedding(dim=768)
    """

    def __init__(
        self,
        dim: int = 768,
    ):
        self.token = nn.Parameter(torch.zeros(dim))

    def forward(self, *_, **__) -> Float[Tensor, " d"]:
        return self.token
