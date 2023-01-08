import torch
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


class PositionEmbed(MixModule):
    def __init__(
        self,
        seq_length: int = 197,
        dim: int = 1024,
    ):
        self.pos_embed = nn.Parameter(torch.randn(seq_length, dim)) * 0.02

    def forward(self, *_) -> Float[Tensor, "n d"]:
        return self.pos_embed
