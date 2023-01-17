import pytest

from torchmix import (
    MLP,
    Add,
    Attach,
    Attention,
    Component,
    Extract,
    PatchEmbed,
    PositionEmbed,
    PreNorm,
    Repeat,
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
                    MLP(
                        dim=1024,
                        expansion_factor=4,
                        act_layer=nn.GELU(),
                    ),
                    dim=1024,
                ),
                PreNorm(
                    Attention(
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
                        MLP(
                            expansion_factor=4,
                            act_layer=nn.GELU(),
                        ),
                    ),
                    PreNorm(
                        Attention(
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
