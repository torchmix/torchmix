import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class RelativePositionBias(Component):
    """Computes the Relative Position Bias for the [Swin Transformer](https://arxiv.org/abs/2103.14030)'s attention mechanism.

    The relative position bias is intended to be added to the attention before
    the softmax is applied. It helps to improve the attention mechanism by
    incorporating the relative positions of the elements in the input.

    Note that the number of parameters is `O(window_size)`, not `O(window_size**2)`.

    Args:
        window_size: The window size for which attention is computed.
        num_heads: Number of heads for the multi-head attention mechanism.

    Examples:
        >>> m = RelativePositionBias(window_size=8, num_heads=8)
        >>> m().shape
        torch.Size([8, 64, 64])
    """

    relative_position_index: Tensor

    def __init__(
        self,
        window_size: int = 8,
        num_heads: int = 8,
    ):
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords_flatten = rearrange(
            torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")),
            "xy h w -> xy (h w)",
        )
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )
        relative_coords = rearrange(relative_coords, "xy h w -> h w xy")
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1

        self.register_buffer("relative_position_index", relative_coords.sum(-1))

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                num_heads, (2 * window_size - 1) * (2 * window_size - 1)
            )
        )

        nn.init.trunc_normal_(self.relative_position_bias_table)

    def forward(self, *_) -> Float[Tensor, "head q k"]:
        """
        Returns:
            Tensor of shape `(num_heads, window_size ** 2, window_size ** 2)` containing the relative position bias for each head and each query-key pair.
        """
        return self.relative_position_bias_table[
            :, self.relative_position_index
        ]
