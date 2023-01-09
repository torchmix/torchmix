from typing import Iterator

from torch import Tensor

from torchmix.core._builds import BuildMode
from torchmix.core._module import Component


class Add(Component):
    """A container that progressively adds the forward results of its children.

    Examples:
        Add(
            nn.Linear(100, 200),
            nn.Linear(100, 200),
            nn.Linear(100, 200)
        )
    """

    build_mode = BuildMode.WITH_ARGS

    def __init__(self, *children: Component):
        for idx, module in enumerate(children):
            self.add_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Component]:
        return iter(self._modules.values())  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        _x: Tensor
        for idx, module in enumerate(self):
            if idx == 0:
                _x = module(x)
            else:
                _x.add_(module(x))
        return _x
