import torchmix
import torchmix.nn as nn  # Wrapped version of torch.nn
from torchmix import (
    Add,
    Attach,
    ChannelMixer,
    Extract,
    PatchEmbed,
    PositionEmbed,
    PreNorm,
    Repeat,
    SelfAttention,
    Token,
)

with torchmix.config(dim=1024):
    model = nn.Sequential(
        Add(
            Attach(
                PatchEmbed(),
                Token(),
            ),
            PositionEmbed(
                seq_length=196 + 1,
            ),
        ),
        Repeat(
            nn.Sequential(
                PreNorm(
                    ChannelMixer(
                        expansion_factor=4,
                        act_layer=nn.GELU.partial(),
                    ),
                ),
                PreNorm(
                    SelfAttention(
                        num_heads=8,
                        head_dim=64,
                    ),
                ),
            ),
            depth=24,
        ),
        Extract(0),
    )

model.export("vit.yaml")
