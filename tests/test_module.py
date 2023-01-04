import pytest
from hydra.utils import instantiate

from torchmix import MixModule, nn


class Dummy(MixModule):
    def __init__(self, a, b=nn.Linear(1, 1)):
        pass


def test_arg_1():
    x = Dummy(3)
    assert x.config.__dict__["a"] == 3
    assert x.config.__dict__["b"].__dict__["in_features"] == 1
    assert x.config.__dict__["b"].__dict__["out_features"] == 1


def test_arg_2():
    x = Dummy(a=3)
    assert x.config.__dict__["a"] == 3
    assert x.config.__dict__["b"].__dict__["in_features"] == 1
    assert x.config.__dict__["b"].__dict__["out_features"] == 1


def test_arg_3():
    x = Dummy(a=3, b=nn.Linear(2, 2))
    assert x.config.__dict__["a"] == 3
    assert x.config.__dict__["b"].__dict__["in_features"] == 2
    assert x.config.__dict__["b"].__dict__["out_features"] == 2


def test_arg_4():
    x = Dummy(3, b=nn.Linear(2, 2))
    assert x.config.__dict__["a"] == 3
    assert x.config.__dict__["b"].__dict__["in_features"] == 2
    assert x.config.__dict__["b"].__dict__["out_features"] == 2


def test_arg_5():
    with pytest.raises(TypeError):
        Dummy(b=nn.Linear(2, 2))


class Dummy2(MixModule):
    def __init__(self, a=1, b=2, c=3):
        pass


def test_arg_6():
    x = Dummy2(2)
    assert x.config.__dict__["a"] == 2
    assert x.config.__dict__["b"] == 2
    assert x.config.__dict__["c"] == 3


def test_arg_7():
    x = Dummy2(2, 3)
    assert x.config.__dict__["a"] == 2
    assert x.config.__dict__["b"] == 3
    assert x.config.__dict__["c"] == 3


def test_arg_8():
    x = Dummy2(2, 3, 4)
    assert x.config.__dict__["a"] == 2
    assert x.config.__dict__["b"] == 3
    assert x.config.__dict__["c"] == 4


def test_arg_9():
    x = Dummy2(2, 3, c=4)
    assert x.config.__dict__["a"] == 2
    assert x.config.__dict__["b"] == 3
    assert x.config.__dict__["c"] == 4


class Dummy3(MixModule):
    def __init__(self, a, b, c, d=4, e=5):
        pass


def test_arg_10():
    x = Dummy3(a=1, b=2, c=3)
    assert x.config.__dict__["a"] == 1
    assert x.config.__dict__["b"] == 2
    assert x.config.__dict__["c"] == 3
    assert x.config.__dict__["d"] == 4
    assert x.config.__dict__["e"] == 5


def test_arg_11():
    x = Dummy3(1, b=2, c=3)
    assert x.config.__dict__["a"] == 1
    assert x.config.__dict__["b"] == 2
    assert x.config.__dict__["c"] == 3
    assert x.config.__dict__["d"] == 4
    assert x.config.__dict__["e"] == 5


def test_arg_12():
    x = Dummy3(1, 2, c=3)
    assert x.config.__dict__["a"] == 1
    assert x.config.__dict__["b"] == 2
    assert x.config.__dict__["c"] == 3
    assert x.config.__dict__["d"] == 4
    assert x.config.__dict__["e"] == 5


def test_arg_13():
    x = Dummy3(1, 2, c=3, e=6)
    assert x.config.__dict__["a"] == 1
    assert x.config.__dict__["b"] == 2
    assert x.config.__dict__["c"] == 3
    assert x.config.__dict__["d"] == 4
    assert x.config.__dict__["e"] == 6


class Dummy4(MixModule):
    def __init__(
        self,
        a=nn.Sequential(
            nn.Linear(1, 2),
            nn.GELU(),
        ),
    ):
        self.a = a


def test_arg_14():
    x = Dummy4()
    assert type(x.a) == nn.Sequential
    assert type(instantiate(x.config).a) == nn.Sequential
    assert x.config.__dict__["a"].__dict__["_args_"][0].__dict__["in_features"] == 1
    assert x.config.__dict__["a"].__dict__["_args_"][0].__dict__["out_features"] == 2
