import torch
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


class Token(MixModule):
    """A single token

    Example:
        >>> model = Token(1024)
        >>> model().shape
        torch.Size([1024])
    """

    def __init__(
        self,
        dim: int = 1024,
    ):
        self.class_token = nn.Parameter(torch.zeros(dim))

    def forward(self, *_) -> Float[Tensor, " d"]:
        return self.class_token
