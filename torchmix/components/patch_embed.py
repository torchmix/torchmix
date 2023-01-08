from jaxtyping import Float
from torch import Tensor, nn

from torchmix.core._module import MixModule
from torchmix.third_party.einops import Rearrange


class PatchEmbed(MixModule):
    """A layer that convert image into patch embeddings.

    Example:
        >>> model = PatchEmbed(16, 3, 768)
        >>> inputs = torch.randn(32, 3, 224, 224)
        >>> model(inputs).shape
        torch.Size([32, 196, 768])
    """

    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        dim: int = 768,
    ):
        # # einops.EinopsError: Ellipsis is not supported in EinMix (right now)
        # self.proj = EinMix(
        #     "... c (h ph) (w pw) -> ... (h w) d",
        #     weight_shape="c ph pw d",
        #     bias_shape="d",
        #     ph=patch_size,
        #     pw=patch_size,
        #     c=channels,
        #     d=dim,
        # )

        self.proj = nn.Sequential(
            Rearrange(
                "... c (h ph) (w pw) -> ... (h w) (c ph pw)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(patch_size * patch_size * channels, dim),
        )

    def forward(self, x: Float[Tensor, "... c h w"]) -> Float[Tensor, "... n d"]:
        return self.proj(x)
