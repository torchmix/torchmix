import random

from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import Component


class Dropout(Component):
    """A dropout layer that wraps a given `Component`.

    Examples:
        Dropout(block=nn.Linear(128, 256), p=0.2)
    """

    def __init__(self, block: Component, p: float = 0.1):
        self.block = block
        self.p = p

    def forward(self, x: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        x = self.block(x)
        x = nn.functional.dropout(
            x,
            p=self.p,
            training=self.training,
            inplace=True,
        )
        return x


class StochasticDepth(Component):
    """A stochastic depth layer that wraps a given `Component`.

    Examples:
        StochasticDepth(block=nn.Linear(128, 256), p=0.2)
    """

    def __init__(self, block: Component, p: float = 0.1):
        self.block = block
        self.p = p

    def forward(self, x: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        if self.p == 0.0 or not self.training:
            return x

        if self.p > random.random():
            return x

        return self.block(x) / (1 - self.p)


class DropPath(Component):
    """A droppath layer that wraps a given `Component`.

    Can be understood as [stochastic depth](/components/stochasticdepth) per sample.

    Examples:
        DropPath(block=nn.Linear(128, 256), p=0.2)
    """

    def __init__(self, block: Component, p: float = 0.1):
        self.block = block
        self.p = p

    def forward(self, x: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        if self.p == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(1 - self.p)
        random_tensor.div_(1 - self.p)
        return self.block(x) * random_tensor
