from torch import Tensor

from torchmix import nn
from torchmix.core._module import Component


class PostNorm(Component):
    """Apply Post-Layer normalization to a block

    Example:
        PreNorm(
            nn.Sequential(
                nn.Linear(100, 200),
                nn.GELU(),
                nn.Linear(200, 100)
            ),
            dim=100
        )
    """

    def __init__(self, block: Component, dim: int = 1024):
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.block(x))


class PreNorm(Component):
    """Apply Pre-Layer normalization to a block

    Examples:
        PreNorm(
            nn.Sequential(
                nn.Linear(100, 200),
                nn.GELU(),
                nn.Linear(200, 100)
            ),
            dim=100
        )
    """

    def __init__(self, block: Component, dim: int = 1024):
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(self.norm(x))
