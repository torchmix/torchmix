# Welcome to torchmix!

`torchmix` is a library that provides a collection of PyTorch modules that aims to make your code more efficient and modular. In this example, we'll demonstrate how to use the config context manager to set common arguments that will be broadcasted to every component.

### Using the `config` Context Manager

Using `torchmix`'s `config` context manager allows you to specify common arguments that will be applied to every component within its scope. This can be especially useful for reducing repetitive code when creating deep learning models with many layers that have common arguments.

For example, consider the following code:

```python
nn.Sequential(
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
        depth=24,
    ),
    Extract(0),
)
```

Rather than specifying the `dim` argument for each layer, you can use the `config` context manager to set it globally and have it automatically applied to every component:

```python
with torchmix.config(dim=1024):
    nn.Sequential(
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
            depth=24,
        ),
        Extract(0),
    )
```

While this pattern can improve the concision of your model definition, it may also increase the interdependence between components.
