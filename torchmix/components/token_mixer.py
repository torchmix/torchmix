from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._module import MixModule
from torchmix.third_party.einops import EinMix


class TokenMixer(MixModule):
    def __init__(
        self,
        act_layer: Partial[MixModule],
        seq_length: int = 196,
        expansion_factor: float = 0.5,
    ):
        self.block = nn.Sequential(
            EinMix(
                # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
                "b n_in d -> b n_out d",
                weight_shape="n_in n_out",
                bias_shape="n_out",
                n_in=seq_length,
                n_out=int(seq_length * expansion_factor),
            ),
            act_layer(),
            EinMix(
                # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
                "b n_out d -> b n_in d",
                weight_shape="n_out n_in",
                bias_shape="n_in",
                n_in=seq_length,
                n_out=int(seq_length * expansion_factor),
            ),
        )

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... n d"]:
        return self.block(x)
