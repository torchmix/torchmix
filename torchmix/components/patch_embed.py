from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule
from torchmix.third_party.einops import EinMix


class PatchEmbed(MixModule):
    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        dim: int = 768,
    ):
        self.projection = EinMix(
            # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
            "b c (h ph) (w pw) -> b (h w) d",
            weight_shape="c ph pw d",
            bias_shape="d",
            ph=patch_size,
            pw=patch_size,
            c=channels,
            d=dim,
        )

    def forward(self, x: Float[Tensor, "... c h w"]) -> Float[Tensor, "... n d"]:
        return self.projection(x)
