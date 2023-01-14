import math

from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component
from torchmix.third_party.einops import Rearrange


class PatchMerging(Component):
    """A patch merging from Swin-Transformer

    Examples:
        PatchMerging(dim=96)
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
        _batch_size, _seq_length, _dim = x.shape

        x = self.merge(
            h=round(math.sqrt(_seq_length / 4)),
            w=round(math.sqrt(_seq_length / 4)),
        )(x)

        return self.proj(x)
