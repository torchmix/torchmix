import torch
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class Token(Component):
    """A single token

    Examples:
        Token(dim=768)
    """

    def __init__(
        self,
        dim: int = 768,
    ):
        self.class_token = nn.Parameter(torch.zeros(dim))

    def forward(self, *_) -> Float[Tensor, " d"]:
        return self.class_token
