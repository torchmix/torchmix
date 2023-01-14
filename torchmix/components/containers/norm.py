from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class PostNorm(Component):
    """Apply Post-Layer normalization to a children

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

    def __init__(self, children: Component, dim: int = 1024):
        self._children = children
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self._children(x))


class PreNorm(Component):
    """Apply Pre-Layer normalization to a children

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

    def __init__(self, children: Component, dim: int = 1024):
        self._children = children
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self._children(self.norm(x))
