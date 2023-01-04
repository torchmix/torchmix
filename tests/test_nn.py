import pytest
from hydra_zen.typing import Partial

from torchmix import MixModule, nn


class CustomModule1(MixModule):
    def __init__(self, dim: int = 100):
        self.stem = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                )
            ]
        )

    def forward(self, x):
        return self.stem(x)


class CustomModule2(MixModule):
    def __init__(self, partial_module: Partial[MixModule]):
        self.module = partial_module()

    def forward(self, x):
        return self.module(x)


testdata = [
    (
        nn.Sequential(
            *[
                nn.Sequential(
                    nn.LayerNorm(100),
                    nn.Linear(100, 100),
                    nn.GELU(),
                    nn.Linear(100, 100),
                )
                for _ in range(10)
            ]
        ),
        204000,
    ),
    (nn.Linear(300, 400), 120400),
    (nn.Conv2d(3, 128, kernel_size=3), 3584),
    (nn.LayerNorm(100), 200),
    (nn.GELU(approximate="tanh"), 0),
    (nn.ReLU(inplace=True), 0),
    (nn.BCELoss(), 0),
    (nn.Dropout(0.4), 0),
    (CustomModule1(dim=100), 20400),
    (CustomModule2(nn.Linear.partial(100, 100)), 10100),
]


@pytest.mark.parametrize("module,num_parameters", testdata)
def test_export(module: MixModule, num_parameters: int):
    assert sum(p.numel() for p in module.parameters()) == num_parameters
