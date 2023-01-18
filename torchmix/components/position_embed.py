import torch
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class PositionEmbed(Component):
    """Learnable positional embeddings

    Examples:
        PositionEmbed(seq_len=196, dim=768)
    """

    def __init__(
        self,
        seq_len: int = 196,
        dim: int = 768,
    ):
        self.pos_embed = nn.Parameter(torch.randn(seq_len, dim)) * 0.02

    def forward(self, *_) -> Float[Tensor, "n d"]:
        return self.pos_embed
