from enum import Enum, auto

from hydra_zen import make_custom_builds_fn


class BuildMode(Enum):
    WITH_ARGS = auto()
    WITHOUT_ARGS = auto()


builds = make_custom_builds_fn(
    populate_full_signature=True,
)
