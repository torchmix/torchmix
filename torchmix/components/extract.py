from typing import Optional, Union

from jaxtyping import Float
from torch import Tensor

from torchmix.core._component import Component


class Extract(Component):
    """Extract layer that selects a single token from an input tensor.

    This layer selects a single token from the input tensor and returns it
    as the output tensor. The token to be extracted is specified by its index.

    Args:
        index (int): Index of the token to be extracted from the input tensor.

    Examples:
        Extract(index=0)
    """

    def __init__(self, start: int, stop: Optional[int] = None):
        self.index = slice(start, stop) if stop else start

    def forward(
        self, x: Float[Tensor, "... n d"]
    ) -> Union[Float[Tensor, "... d"], Float[Tensor, "... n_out d"]]:
        return x[..., self.index, :]
