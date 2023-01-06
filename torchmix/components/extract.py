from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule


class Extract(MixModule):
    def __init__(self, index: int):
        self.index = index

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
        return x[:, self.index, :]
