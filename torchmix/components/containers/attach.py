from einops import pack, repeat
from torch import Tensor

from torchmix.nn import Sequential


class Attach(Sequential):
    """Concatenate results over penultimate dimension.

    Example:
        Attach(
            Token(dim=1024),
            PatchEmbed(dim=1024),
        )
    """

    def forward(self, x: Tensor) -> Tensor:
        tokens = []
        for idx, module in enumerate(self):
            _token: Tensor = module(x)
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
            elif _token.dim() >= 3:
                tokens.append(_token)

        x, _packed_shape = pack(tokens, "b * c")

        return x
