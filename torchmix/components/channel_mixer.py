from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule
from torchmix.third_party.einops import EinMix


class ChannelMixer(MixModule):
    def __init__(
        self,
        act_layer: Partial[MixModule],
        dim: int = 1024,
        expansion_factor: float = 4,
    ):
        self.block = nn.Sequential(
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=dim,
                d_out=int(dim * expansion_factor),
            ),
            act_layer(),
            EinMix(
                "b n d_out -> b n d_in",
                weight_shape="d_out d_in",
                bias_shape="d_in",
                d_in=dim,
                d_out=int(dim * expansion_factor),
            ),
        )

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        return self.block(x)
