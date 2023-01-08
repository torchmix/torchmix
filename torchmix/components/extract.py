from jaxtyping import Float
from torch import Tensor

from torchmix.core._module import MixModule


class Extract(MixModule):
    """Extract layer that selects a single token from an input tensor.

    This layer selects a single token from the input tensor and returns it
    as the output tensor. The token to be extracted is specified by its index.

    Args:
        index (int): Index of the token to be extracted from the input tensor.

    Returns:
        Float[Tensor, "... d"]: Output tensor with the same leading dimensions
            as the input tensor, but with a single token representing the
            extracted token.

    Examples:
        >>> model = Extract(0)
        >>> inputs = torch.randn(32, 196, 1024)
        >>> model(inputs).shape
        torch.Size([32, 1024])
    """

    def __init__(self, index: int):
        self.index = index

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... d"]:
        return x[..., self.index, :]
