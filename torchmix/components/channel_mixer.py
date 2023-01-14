from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component


class ChannelMixer(Component):
    """Channel mixer that performs token-wise transformations on an input tensor.

    This layer applies two EinMix layers in sequence to perform token-wise
    transformations on the input tensor. The first EinMix layer expands
    the number of channels by a specified expansion factor, and the second
    EinMix layer reduces the number of channels back to its original value.

    An activation layer can be inserted between the two EinMix layers.

    Args:
        act_layer: Activation layer to be inserted
            between the two EinMix layers.
        dim: Number of channels in the input tensor. Defaults to 1024.
        expansion_factor: Factor by which to expand
            the number of channels in the first EinMix layer. Defaults to 4.

    Examples:
        ChannelMixer(act_layer=nn.GELU.partial(), dim=768, expansion_factor=4)
    """

    def __init__(
        self,
        act_layer: Partial[Component],
        dim: int = 768,
        expansion_factor: float = 4,
    ):
        self.block = nn.Sequential(
            nn.Linear(dim, int(dim * expansion_factor)),
            act_layer(),
            nn.Linear(int(dim * expansion_factor), dim),
        )

    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        """Perform channel-wise transformations on the input tensor.

        Args:
            x (Float[Tensor, "... d"]): Input tensor with any number of
                leading dimensions and a trailing dimension representing
                the number of channels.

        Returns:
            Float[Tensor, "... d"]: Output tensor with the same dimensions
                as the input tensor, but with the channels transformed by
                the channel mixer.
        """
        return self.block(x)
