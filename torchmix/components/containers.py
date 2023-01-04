from typing import Iterator

from einops import pack, repeat
from torch import Tensor

from torchmix import nn
from torchmix.core._builds import BuildMode
from torchmix.core._module import MixModule

__all__ = ["Attach", "Add", "Repeat", "PreNorm", "PostNorm"]


class Attach(MixModule):
    build_mode = BuildMode.WITH_ARGS

    def __init__(self, *blocks: MixModule):
        for idx, block in enumerate(blocks):
            self.add_module(str(idx), block)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[MixModule]:
        return iter(self._modules.values())  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        tokens = []
        for idx, block in enumerate(self):
            _token: Tensor = block(x)
            if idx == 0:
                if _token.dim() != 3:
                    raise ValueError(
                        "Invalid tensor dimensions: "
                        f"expected 3 dimensions, got {_token.dim()}"
                    )
                batch_size, *_ = _token.shape

            if _token.dim() == 1:
                tokens.append(repeat(_token, "d -> b d", b=batch_size))
            elif _token.dim() == 2:
                tokens.append(repeat(_token, "n d -> b n d", b=batch_size))
            elif _token.dim() == 3:
                tokens.append(_token)
            else:
                raise ValueError(
                    "Invalid tensor dimensions: "
                    f"expected at most 3 dimensions, got {_token.dim()}"
                )

        x, _packed_shape = pack(tokens, "b * c")

        return x


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


class Repeat(MixModule):
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


class PreNorm(MixModule):
    def __init__(self, block: MixModule, dim: int = 1024):
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(self.norm(x))


class PostNorm(MixModule):
    def __init__(self, block: MixModule, dim: int = 1024):
        self.block = block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x + self.block(x))
