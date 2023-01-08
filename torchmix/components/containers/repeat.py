from typing import Iterator

from torch import Tensor

from torchmix.core._module import MixModule


class Repeat(MixModule):
    """Repeat given module.

    Examples:
        Repeat(
            nn.Sequential(
                nn.Linear(100, 200),
                nn.GELU(),
                nn.Linear(200, 100)
            ),
            depth=12
        )
    """

    def __init__(
        self,
        block: MixModule,
        depth: int = 8,
    ) -> None:
        for idx in range(depth):
            self.add_module(
                str(idx),
                block.instantiate(),
            )

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[MixModule]:
        return iter(self._modules.values())  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        for module in self:
            x = module(x)
        return x
