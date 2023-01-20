import pytest

from torchmix import (
    Add,
    Attach,
    Attention,
    ClassEmbedding,
    ClassPool,
    Component,
    Feedforward,
    PatchEmbedding,
    PositionalEmbedding,
    PreNorm,
    Repeat,
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
                ClassEmbedding(dim=1024),
                PatchEmbedding(dim=1024),
            ),
            PositionalEmbedding(
                seq_len=196 + 1,
                dim=1024,
            ),
        ),
        Repeat(
            nn.Sequential(
                PreNorm(
                    Feedforward(
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
        ClassPool(),
    )

    with config(dim=1024):
        model_2 = nn.Sequential(
            Add(
                Attach(
                    ClassEmbedding(),
                    PatchEmbedding(),
                ),
                PositionalEmbedding(
                    seq_len=196 + 1,
                ),
            ),
            Repeat(
                nn.Sequential(
                    PreNorm(
                        Feedforward(
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
            ClassPool(),
        )

    # assert model_1.config == model_2.config  # TODO
    assert helpers.is_module_equal(model_1, model_2)
