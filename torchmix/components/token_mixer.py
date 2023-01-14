from typing import Optional

from hydra_zen.typing import Partial
from jaxtyping import Float
from torch import Tensor

from torchmix import nn
from torchmix.core._component import Component
from torchmix.third_party.einops import Rearrange

from .containers.drop import Dropout


class TokenMixer(Component):
    """Token mixer layer from MLP-Mixer

    Examples:
        TokenMixer(
            act_layer=nn.GELU.partial(), seq_length=196, expansion_factor=0.5, p=0.1
        )
    """

    def __init__(
        self,
        act_layer: Partial[Component],
        seq_length: int = 196,
        expansion_factor: float = 0.5,
        p: Optional[float] = 0.1,
    ):
        # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
        # self.block = nn.Sequential(
        #     EinMix(
        #         "b n_in d -> b n_out d",
        #         weight_shape="n_in n_out",
        #         bias_shape="n_out",
        #         n_in=seq_length,
        #         n_out=int(seq_length * expansion_factor),
        #     ),
        #     act_layer(),
        #     EinMix(
        #         "b n_out d -> b n_in d",
        #         weight_shape="n_out n_in",
        #         bias_shape="n_in",
        #         n_in=seq_length,
        #         n_out=int(seq_length * expansion_factor),
        #     ),
        # )
        self.block = nn.Sequential(
            Rearrange(".. n d -> ... d n"),
            Dropout(
                nn.Linear(seq_length, int(seq_length * expansion_factor)),
                p=self.p,
            ),
            act_layer(),
            Dropout(
                nn.Linear(int(seq_length * expansion_factor), seq_length),
                p=self.p,
            ),
            Rearrange("... d n -> ... n d"),
        )

    def forward(self, x: Float[Tensor, "... n d"]) -> Float[Tensor, "... n d"]:
        return self.block(x)
