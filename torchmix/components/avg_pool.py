from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule
from torchmix.third_party.einops import Reduce


class AvgPool(MixModule):
    def __init__(self):
        self.pool = Reduce("b n d -> b d", reduction="mean")

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
        return self.pool(x)
