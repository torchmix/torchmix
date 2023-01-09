import pytest

from torchmix import (
    Add,
    Attach,
    ChannelMixer,
    Component,
    Extract,
    PatchEmbed,
    PositionEmbed,
    PreNorm,
    Repeat,
    SelfAttention,
    Token,
    nn,
)
from torchmix.core._context import config


class Dummy(Component):
    def __init__(self, dim=1):
        self.dim = dim


def test_config_context_1():
    with config(dim=2):
        x = Dummy()
    assert x.dim == 2


@pytest.mark.slow
def test_config_context_2(helpers):
    model_1 = nn.Sequential(
        Add(
            Attach(
                Token(dim=1024),
                PatchEmbed(dim=1024),
            ),
            PositionEmbed(
                seq_length=196 + 1,
                dim=1024,
            ),
        ),
        Repeat(
            nn.Sequential(
                PreNorm(
                    ChannelMixer(
                        dim=1024,
                        expansion_factor=4,
                        act_layer=nn.GELU.partial(),
                    ),
                    dim=1024,
                ),
                PreNorm(
                    SelfAttention(
                        dim=1024,
                        num_heads=8,
                        head_dim=64,
                    ),
                    dim=1024,
                ),
            ),
            depth=2,
        ),
        Extract(0),
    )

    with config(dim=1024):
        model_2 = nn.Sequential(
            Add(
                Attach(
                    Token(),
                    PatchEmbed(),
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
                depth=2,
            ),
            Extract(0),
        )

    # assert model_1.config == model_2.config  # TODO
    assert helpers.is_module_equal(model_1, model_2)
