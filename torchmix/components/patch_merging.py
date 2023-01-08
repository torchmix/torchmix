import math

from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule
from torchmix.third_party.einops import Rearrange


class PatchMerging(MixModule):
    """A patch merging from Swin-Transformer

    Example:
        >>> model = PatchMerging(96)
        >>> inputs = torch.randn(32, 56*56, 96)
        >>> model(inputs).shape
        torch.Size([32, 784, 192])
    """

    def __init__(self, dim: int = 96):
        self.merge = Rearrange.partial(
            "... (h ph w pw) d -> ... (h w) (ph pw d)",
            ph=2,
            pw=2,
            h=None,
            w=None,
        )

        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 4),
            # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
            # EinMix(
            #     "... d_in -> ... d_out",
            #     weight_shape="d_in d_out",
            #     bias_shape="d_out",
            #     d_in=dim * 4,
            #     d_out=dim * 2,
            # ),
            nn.Linear(dim * 4, dim * 2),
        )

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... n/4 d*2"]:
        _batch_size, _seq_length, _dim = x.shape

        x = self.merge(
            h=round(math.sqrt(_seq_length / 4)),
            w=round(math.sqrt(_seq_length / 4)),
        )(x)

        return self.proj(x)
