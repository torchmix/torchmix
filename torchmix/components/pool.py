import math
from typing import Optional, Union

from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core.component import Component
from torchmix.third_party.einops import Reduce


class AvgPool(Component):
    """Average pooling layer that averages over the penultimate dimension of an input tensor.

    Examples:
        AvgPool()
    """

    def __init__(self):
        self.pool = Reduce("... n d -> ... d", reduction="mean")

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... d"]:
        return self.pool(x)


class ClassPool(Component):
    """Extracts the representation of CLS Token

    Args:
        index: Index of the CLS token to be extracted.

    Examples:
        CLSTokenPool(0)
    """

    def __init__(self, start: int = 0, stop: Optional[int] = None):
        self.index = slice(start, stop) if stop else start

    def forward(
        self, x: Float[Tensor, "... n_in d"]
    ) -> Union[Float[Tensor, "... d"], Float[Tensor, "... n_out d"]]:
        return x[..., self.index, :]


class PatchMerge(Component):
    """Patch merging layer from [Swin-Transformer](https://arxiv.org/abs/2103.14030).

    Examples:
        PatchMerging(dim=96, expansion_factor=2.0)
    """

    def __init__(
        self,
        dim: int = 96,
        expansion_factor: float = 2.0,
    ):
        self.merge = Rearrange.partial(
            "... (h ph w pw) d -> ... (h w) (ph pw d)",
            ph=2,
            pw=2,
            h=None,
            w=None,
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, int(dim * expansion_factor)),
        )

    def forward(
        self, x: Float[Tensor, "... n d_in"]
    ) -> Float[Tensor, "... n/4 d_out"]:
        _batch_size, _seq_len, _dim = x.shape

        x = self.merge(
            h=round(math.sqrt(_seq_len / 4)),
            w=round(math.sqrt(_seq_len / 4)),
        )(x)

        return self.proj(x)
