from torchmix.core._builds import BuildMode
from torchmix.core._component import Component
from torchmix.nn import Sequential


class Repeat(Sequential):
    """Repeat the given module `depth` times.

    `Repeat` creates multiple copies of the `children` and
    applies them sequentially. Every copy of `children` will
    be re-instantiated based on its configuration. The input
    and output shapes of the `children` must be the same in
    order to be applied in this way.

    Args:
        children (Component): The module to be repeated.
        depth (int): The number of copies of `children` to create.

    Examples:
        Repeat(
            nn.Sequential(
                nn.Linear(100, 200),
                nn.GELU(),
                nn.Linear(200, 100)
            ),
            depth=12
        )
    """

    build_mode = BuildMode.WITHOUT_ARGS

    def __init__(
        self,
        children: Component,
        depth: int = 8,
    ) -> None:
        self.add_module(
            str(0),
            children,
        )

        for idx in range(1, depth):
            self.add_module(
                str(idx),
                children.instantiate(),
            )
