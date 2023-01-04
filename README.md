<h1 align="center">torchmix</h1>

<h3 align="center">The missing component library for PyTorch</h3>

<br />

Welcome to `torchmix`, a collection of PyTorch modules that aims to make your code more efficient and modular. We've included a range of operations, from basic ones like `Repeat` and `Add`, to more complex ones like `WindowAttention` in the [Swin-Transformer](https://arxiv.org/abs/2103.14030). Our goal is to make it easy for you to use these various operations with minimal code, so you can focus on building your project rather than writing boilerplate.

We've designed `torchmix` to be as user-friendly as possible. Each implementation is kept minimal and easy to understand, using [`einops`](https://github.com/arogozhnikov/einops) to avoid confusing tensor manipulation (such as `permute`, `transpose`, and `reshape`) and [`jaxtyping`](https://github.com/google/jaxtyping) to clearly document the shapes of the input and output tensors. This means that you can use `torchmix` with confidence, knowing that the components you're working with are clean and reliable.

**Note: `torchmix` is a prototype that is currently in development and has not been tested for production use. The API may change at any time.**

<br />

## Install

To use `torchmix`, you will need to have `torch` already installed on your environment.

```sh
pip install torchmix
```

## Usage

To use `torchmix`, simply import the desired components:

```python
import torchmix.nn as nn  # Wrapped version of torch.nn
from torchmix import (
    Add,
    Attach,
    AvgPool,
    ChannelMixer,
    Extract,
    PatchEmbed,
    PositionEmbed,
    PreNorm,
    Repeat,
    SelfAttention,
    Token,
)
```

You can simply compose this components to build more complex architecture:

```python
# ViT with CLS Token attached
vit_cls = nn.Sequential(
    Add(
        Attach(
            PatchEmbed(dim=1024),
            Token(dim=1024),
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
        depth=24,
    ),
    Extract(0),
)

# ViT with average pooling
vit_avg = nn.Sequential(
    Add(
        PatchEmbed(dim=1024),
        PositionEmbed(
            seq_length=196,
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
        depth=24,
    ),
    AvgPool(),
)
```

### Integration with [Hydra](https://hydra.cc/)

Reproducibility is important, so it is always a good idea to manage the configurations of your models. However, manually writing the configurations for complex, deeply nested PyTorch modules can be time-consuming and result in confusing and hard-to-maintain code. This is often because the parent class is responsible for accepting and passing along the parameters to its children classes, leading to a large number of arguments and strong coupling between the parent and children classes.

`torchmix` was designed to address this issue by [**auto-magically**](https://github.com/mit-ll-responsible-ai/hydra-zen) generating the full configuration of a PyTorch modules **simply by instantiating them**, regardless of how deeply they are nested. This makes it easy to integrate your favorite modules into [`hydra`](https://hydra.cc/) ecosystem. This **instantiate-for-configuration** pattern also promotes the direct injection of dependencies, leading to the creation of loosely-coupled components and more declarative and intuitive code.

In other words, getting a configuration with `torchmix` is practically effortless - it's just there for you:

```python
from torchmix import nn

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.Dropout(0.1),
    nn.GELU(),
    nn.Linear(4096, 1024),
    nn.Dropout(0.1),
)

model.config  # DictConfig which contains full signatures 🤯
```

You can then store the configuration in the [`hydra`'s `ConfigStore`](https://hydra.cc/docs/tutorials/structured_config/config_store/) using:

```python
model.store(group="model", name="mlp")
```

Alternatively, you can export it to a YAML file if you want:

```python
model.export("mlp.yaml")
```

This will generate the following configuration:

```yaml
_target_: torchmix.nn.Sequential
_args_:
  - _target_: torchmix.nn.Linear
    in_features: 1024
    out_features: 4096
    bias: true
    device: null
    dtype: null
  - _target_: torchmix.nn.Dropout
    p: 0.1
    inplace: false
  - _target_: torchmix.nn.GELU
    approximate: none
  - _target_: torchmix.nn.Linear
    in_features: 4096
    out_features: 1024
    bias: true
    device: null
    dtype: null
  - _target_: torchmix.nn.Dropout
    p: 0.1
    inplace: false
```

You can always get the actual PyTorch module from its configuration using [`hydra`'s `instantiate`.](https://hydra.cc/docs/advanced/instantiate_objects/overview/)

To create custom modules with this functionality, simply subclass `MixModule` and define your module as you normally would:

```python
from torchmix import MixModule

class CustomModule(MixModule):
    def __init__(self, num_heads, dim, depth):
        pass

custom_module = CustomModule(16, 768, 12)
custom_module.store(group="model", name="custom")
```

## Examples

For more information on using `torchmix`'s functionality, see the following examples:

- [Using the `no_parameters` Context Manager](https://github.com/torchmix/torchmix/tree/main/examples/no_parameters)
- [Using the `partial` Method](https://github.com/torchmix/torchmix/tree/main/examples/partial)

## Documentation

Documentation is currently in progress. Please stay tuned! 🚀

## Contributing

The development of `torchmix` is an open process, and we welcome any contributions or suggestions for improvement. If you have ideas for new components or ways to enhance the library, feel free to open an issue or start a discussion. We welcome all forms of feedback, including criticism and suggestions for significant design changes. Please note that `torchmix` is currently in the early stages of development and any contributions should be considered experimental. Thank you for your support of `torchmix`!

## License

`torchmix` is licensed under the MIT License.
