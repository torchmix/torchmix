from einops import rearrange

from torchmix import nn
from torchmix.core.component import Component


class FeedforwardPlugin(Component):
    """Base class for all plugins for Feedforward layers.

    Examples:
        class CustomPlugin(FeedforwardPlugin): ...
    """

    def pre_proj_in(self, x):
        return x

    def post_proj_in(self, x):
        return x

    def pre_act(self, x):
        return x

    def post_act(self, x):
        return x

    def pre_proj_out(self, x):
        return x

    def post_proj_out(self, x):
        return x


class DropProjectionIn(FeedforwardPlugin):
    """Apply dropout after first linear layer.

    Examples:
        Feedforward(
            dim=768,
            plugins=[
                DropProjectionIn(p=0.1),
            ],
        )
    """

    def __init__(self, p: float = 0.1):
        self.drop = nn.Dropout(p)

    def post_proj_in(self, x):
        return self.drop(x)


class DropActivation(FeedforwardPlugin):
    """Apply dropout after activation layer.

    Examples:
        Feedforward(
            dim=768,
            plugins=[
                DropActivation(p=0.1),
            ],
        )
    """

    def __init__(self, p: float = 0.1):
        self.drop = nn.Dropout(p)

    def post_act(self, x):
        return self.drop(x)


class DropProjectionOut(FeedforwardPlugin):
    """Apply dropout after activation layer.

    Examples:
        Feedforward(
            dim=768,
            plugins=[
                DropProjectionOut(p=0.1),
            ],
        )
    """

    def __init__(self, p: float = 0.1):
        self.drop = nn.Dropout(p)

    def post_proj_out(self, x):
        return self.drop(x)


class Transpose(FeedforwardPlugin):
    """Applies Feedforward to penultimate dimension.

    This plugin can be used to implement token-mixer from [Feedforward-Mixer](https://arxiv.org/abs/2105.01601)

    Examples:
        Feedforward(
            dim=196,
            plugins=[
                Transpose(),
            ],
        )

    """

    def pre_proj_in(self, x):
        return rearrange(x, "... n d -> ... d n")

    def post_proj_out(self, x):
        return rearrange(x, "... d n -> ... n d")
