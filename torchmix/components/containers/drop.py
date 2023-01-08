import random

from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


class Dropout(MixModule):
    """A dropout layer that wraps a given MixModule.

    Example:
        >>> model = Dropout(nn.Linear(128, 256), p=0.2)
        >>> inputs = torch.randn(32, 128)
        >>> model(inputs).shape
        torch.Size([32, 256])
    """

    def __init__(self, block: MixModule, p: float = 0.1):
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


class StochasticDepth(MixModule):
    """A stochastic depth layer that wraps a given MixModule.

    Example:
        >>> model = StochasticDepth(nn.Linear(128, 256), p=0.2)
        >>> inputs = torch.randn(32, 128)
        >>> model(inputs).shape
        torch.Size([32, 256])
    """

    def __init__(self, block: MixModule, p: float = 0.1):
        self.block = block
        self.p = p

    def forward(self, x: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        if self.p == 0.0 or not self.training:
            return x

        if self.p > random.random():
            return x

        return self.block(x) / (1 - self.p)


class DropPath(MixModule):
    """A droppath layer that wraps a given MixModule.

    Can be understood as stochastic depth per sample.

    Example:
        >>> model = DropPath(nn.Linear(128, 256), p=0.2)
        >>> inputs = torch.randn(32, 128)
        >>> model(inputs).shape
        torch.Size([32, 256])
    """

    def __init__(self, block: MixModule, p: float = 0.1):
        self.block = block
        self.p = p

    def forward(self, x: Float[Tensor, " *shape"]) -> Float[Tensor, " *shape"]:
        if self.p == 0.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(1 - self.p)
        random_tensor.div_(1 - self.p)
        return self.block(x) * random_tensor
