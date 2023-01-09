import torch
from hydra_zen.typing import Partial

from torchmix import Component, nn


class CustomModule(Component):
    def __init__(
        self,
        proj_layer: Partial[Component],
        act_layer: Partial[Component],
    ):
        self.proj = proj_layer
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.proj(x))


custom_module = CustomModule(
    proj_layer=nn.Linear.partial(100, 100),
    act_layer=nn.GELU.partial(),
)
custom_module.store(group="model", name="linear").export()
