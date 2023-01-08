from typing import Iterator

from torch import Tensor

from torchmix.core._builds import BuildMode
from torchmix.core._module import MixModule


class Add(MixModule):
    build_mode = BuildMode.WITH_ARGS

    def __init__(self, *blocks: MixModule):
        for idx, block in enumerate(blocks):
            self.add_module(str(idx), block)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[MixModule]:
        return iter(self._modules.values())  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        _x: Tensor
        for idx, module in enumerate(self):
            if idx == 0:
                _x = module(x)
            else:
                _x.add_(module(x))
        return _x
