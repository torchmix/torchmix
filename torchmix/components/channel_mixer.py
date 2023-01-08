from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule


class ChannelMixer(MixModule):
    """Channel mixer that performs token-wise transformations on
    an input tensor.

    This layer applies two EinMix layers in sequence to perform token-wise
    transformations on the input tensor. The first EinMix layer expands
    the number of channels by a specified expansion factor, and the second
    EinMix layer reduces the number of channels back to its original value.

    An activation layer can be inserted between the two EinMix layers.

    Args:
        act_layer (Partial[MixModule]): Activation layer to be inserted
            between the two EinMix layers.
        dim (int): Number of channels in the input tensor.
            Default: 1024.
        expansion_factor (float): Factor by which to expand
            the number of channels in the first EinMix layer. Default: 4.

    Examples:
        >>> channel_mixer = ChannelMixer()
        >>> model = torch.randn(32, 196, 1024)
        >>> model(x).shape
        torch.Size([32, 196, 1024])
    """

    def __init__(
        self,
        act_layer: Partial[MixModule],
        dim: int = 1024,
        expansion_factor: float = 4,
    ):
        # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
        # self.block = nn.Sequential(
        #     EinMix(
        #         "... d_in -> ... d_out",
        #         weight_shape="d_in d_out",
        #         bias_shape="d_out",
        #         d_in=dim,
        #         d_out=int(dim * expansion_factor),
        #     ),
        #     act_layer(),
        #     EinMix(
        #         "... d_out -> ... d_in",
        #         weight_shape="d_out d_in",
        #         bias_shape="d_in",
        #         d_in=dim,
        #         d_out=int(dim * expansion_factor),
        #     ),
        # )
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
