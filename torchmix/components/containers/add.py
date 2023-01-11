from torch import Tensor

from torchmix.nn import Sequential


class Add(Sequential):
    """Cumulatively add the forward results of the given modules in parallel.

    `Add` inherits from the `Sequential` class and applies a list of modules
    in parallel, adding their forward results cumulatively. The output shapes
    of the given modules must be the same or broadcastable in order to be added.

    Args:
        *args: `Component` instances whose forward results will be added cumulatively.

    Examples:
        Add(
            nn.Linear(100, 200),
            nn.Linear(100, 200),
            nn.Linear(100, 200)
        )
    """

    def forward(self, x: Tensor) -> Tensor:
        _x: Tensor
        for idx, module in enumerate(self):
            if idx == 0:
                _x = module(x)
            else:
                _x.add_(module(x))
        return _x
