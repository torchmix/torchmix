import pytest

from torchmix import Component, nn, no_parameters

with no_parameters():
    testdata = [
        nn.Linear(10000, 10000),
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(100, 400),
                    nn.GELU(),
                    nn.Linear(400, 100),
                    nn.LayerNorm(100),
                )
                for _ in range(10)
            ]
        ),
    ]


@pytest.mark.parametrize("module", testdata)
def test_context(module: Component):
    with pytest.raises(AttributeError):
        getattr(module, "_parameters")
