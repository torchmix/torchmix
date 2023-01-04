import os
import tempfile

import pytest
from hydra_zen import instantiate
from hydra_zen.typing import Partial
from omegaconf import OmegaConf

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
    nn.Linear(300, 400),
    nn.Conv2d(3, 128, kernel_size=3),
    nn.LayerNorm(100),
    nn.GELU(approximate="tanh"),
    nn.ReLU(inplace=True),
    nn.BCELoss(),
    nn.Dropout(0.4),
    CustomModule1(dim=100),
    CustomModule2(nn.Linear.partial(100, 100)),
]


@pytest.mark.parametrize("module", testdata)
def test_export(module: MixModule):
    temp_dir = tempfile.TemporaryDirectory()
    file_path = os.path.join(temp_dir.name, "sequential.yaml")
    module.export(file_path)

    assert module.config == OmegaConf.load(file_path)

    temp_dir.cleanup()


@pytest.mark.parametrize("module", testdata)
def test_instantiate(module: MixModule, helpers):
    assert helpers.is_module_equal(module, instantiate(module.config))
