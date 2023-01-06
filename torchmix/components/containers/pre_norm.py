from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


class PreNorm(MixModule):
    def __init__(self, block: MixModule, dim: int = 1024):
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(self.norm(x))
