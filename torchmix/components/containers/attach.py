from typing import Iterator

from einops import pack, repeat
from torch import Tensor

from torchmix.core._builds import BuildMode
from torchmix.core._module import Component


class Attach(Component):
    build_mode = BuildMode.WITH_ARGS

    def __init__(self, *blocks: Component):
        for idx, block in enumerate(blocks):
            self.add_module(str(idx), block)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Component]:
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
