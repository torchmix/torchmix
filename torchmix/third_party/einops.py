from einops.layers._einmix import _EinmixMixin
from einops.layers.torch import EinMix as _EinMix
from einops.layers.torch import Rearrange as _Rearrange
from einops.layers.torch import RearrangeMixin as _RearrangeMixin
from einops.layers.torch import Reduce as _Reduce
from einops.layers.torch import ReduceMixin as _ReduceMixin

from torchmix.core._component import Component


class EinMix(Component, _EinMix):
    __init__ = _EinmixMixin.__init__  # type: ignore


class Rearrange(Component, _Rearrange):
    __init__ = _RearrangeMixin.__init__  # type: ignore


class Reduce(Component, _Reduce):
    __init__ = _ReduceMixin.__init__  # type: ignore
