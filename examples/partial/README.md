# Welcome to torchmix!

`torchmix` is a library that provides a collection of PyTorch modules that aims to make your code more efficient and modular. In this example, we'll show you how to use `torchmix`'s partial method to create partially instantiated `Component` objects and how to use them as arguments for other `Component` objects.

### Using the `partial` Method

The `partial` method allows you to create a partially instantiated version of a `Component`. This means that some of its arguments have not yet been specified. For Examples:

```python
from torchmix import nn

# Create a partially instantiated version of the modules
linear_partial = nn.Linear.partial(100, 100)
gelu_partial = nn.GELU.partial(approximate="tanh")
```

### Using Partially Instantiated Modules as Arguments

You can use a partially instantiated `Component` as an argument for another `Component` just like any other argument. When the second `Component` is instantiated, it will recognize that the partially instantiated `Component` is a `Component` and parse its configurations accordingly.

Here's an example of how to use a partially instantiated `Component` as an argument:

```python
import torch
from torchmix import nn, Component

class CustomModule(Component):
    def __init__(
        self,
        proj_layer: Partial[Component],
        act_layer: Partial[Component],
    ):
        self.proj = proj_layer()
        self.act = act_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.proj(x))

custom_module = CustomModule(
    proj_layer=linear_partial,
    act_layer=gelu_partial,
)
```

Instead, you can provide fully instantiate `Component` and re-`instantiate` from its configuration. This pattern is useful in certain cases, such as when you need to instantiate the same module multiple times.

```python
import torch

from hydra.utils import instantiate
from torchmix import Component, nn


class CustomModule(Component):
    def __init__(
        self,
        proj_layer: Component,
        act_layer: Component,
    ):
        self.proj = proj_layer.instance() # equivalent to instantiate(proj_layer.config)
        self.act = act_layer.instance() # equivalent to instantiate(act_layer.config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.proj(x))

custom_module = CustomModule(
    dim=nn.Linear(100, 100),
    act_layer=nn.GELU(),
)
```

### Storing and Exporting Configurations

Once you've created your `Component` object, you can use the store and export methods to store its configurations in hydra's `ConfigStore` and export them to a YAML file, respectively.

For example:

```python
# Store the custom module's configurations in the ConfigStore and export it to a YAML file
custom_module.store(group="model", name="linear").export()
```

You can view the generated configurations [here](https://github.com/torchmix/torchmix/blob/main/examples/partial/custom.yaml)
