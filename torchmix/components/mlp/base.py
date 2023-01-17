from typing import List

from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component

from .plugin import MLPPlugin


class MLP(Component):
    """Base class for all MLPs.

    Args:
        dim: Dimension size.
        act_layer: Activation layer to be inserted between the two Linear layers.
        expansion_factor: Factor by which to expand `dim` in the first Linear layer.
        plugins: A list of [`MLPPlugin`](/plugins/MLPPlugin)s to use.

    Examples:
        MLP(act_layer=nn.GELU(), dim=768, expansion_factor=4, plugins=[])
    """

    def __init__(
        self,
        dim: int = 768,
        expansion_factor: float = 4,
        act_layer: Component = nn.GELU(),
        plugins: List[MLPPlugin] = [],
    ):
        proj_dim = int(dim * expansion_factor)

        self.proj_in_init(dim, proj_dim)
        self.act = act_layer
        self.proj_out_init(dim, proj_dim)

        self.plugins: List[MLPPlugin] = nn.ModuleList(plugins)

    def proj_in(
        self, x: Float[Tensor, "... d_in"]
    ) -> Float[Tensor, "... d_out"]:
        return self._proj_in(x)

    def proj_in_init(self, dim: int, proj_dim: int):
        self._proj_in = nn.Linear(dim, proj_dim)

    def proj_out(
        self, x: Float[Tensor, "... d_out"]
    ) -> Float[Tensor, "... d_in"]:
        return self._proj_out(x)

    def proj_out_init(self, dim: int, proj_dim: int):
        self._proj_out = nn.Linear(proj_dim, dim)

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        for plugin in self.plugins:
            x = plugin.pre_proj_in(x)
        x = self.proj_in(x)
        for plugin in self.plugins:
            x = plugin.post_proj_in(x)

        for plugin in self.plugins:
            x = plugin.pre_act(x)
        x = self.act(x)
        for plugin in self.plugins:
            x = plugin.post_act(x)

        for plugin in self.plugins:
            x = plugin.pre_proj_out(x)
        x = self.proj_out(x)
        for plugin in self.plugins:
            x = plugin.post_proj_out(x)

        return x
