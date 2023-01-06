from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule
from torchmix.third_party.einops import Reduce


class AvgPool(MixModule):
    """Average pooling layer that averages over the penultimate dimension
    of an input tensor.
    """

    def __init__(self):
        self.pool = Reduce("... n d -> ... d", reduction="mean")

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... d"]:
        return self.pool(x)
