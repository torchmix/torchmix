from jaxtyping import Float
from torch import Tensor

from torchmix.core._component import Component
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
