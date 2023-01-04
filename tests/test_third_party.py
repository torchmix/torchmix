from hydra_zen import to_yaml

from torchmix.third_party.einops import EinMix


def test_einmix1():
    model = EinMix(
        "b n d_in -> b n d_out",
        weight_shape="d_in d_out",
        bias_shape="d_out",
        d_in=1,
        d_out=2,
    )
    print(to_yaml(model.config))
    assert model.config.__dict__["pattern"] == "b n d_in -> b n d_out"
    assert model.config.__dict__["weight_shape"] == "d_in d_out"
    assert model.config.__dict__["bias_shape"] == "d_out"
    assert model.config.__dict__["d_in"] == 1
    assert model.config.__dict__["d_out"] == 2


def test_einmix2():
    model = EinMix(
        "b n d_in -> b n d_out",
        "d_in d_out",
        bias_shape="d_out",
        d_in=1,
        d_out=2,
    )
    print(to_yaml(model.config))
    assert model.config.__dict__["pattern"] == "b n d_in -> b n d_out"
    assert model.config.__dict__["weight_shape"] == "d_in d_out"
    assert model.config.__dict__["bias_shape"] == "d_out"
    assert model.config.__dict__["d_in"] == 1
    assert model.config.__dict__["d_out"] == 2


def test_einmix3():
    model = EinMix(
        "b n d_in -> b n d_out",
        "d_in d_out",
        "d_out",
        d_in=1,
        d_out=2,
    )
    print(to_yaml(model.config))
    assert model.config.__dict__["pattern"] == "b n d_in -> b n d_out"
    assert model.config.__dict__["weight_shape"] == "d_in d_out"
    assert model.config.__dict__["bias_shape"] == "d_out"
    assert model.config.__dict__["d_in"] == 1
    assert model.config.__dict__["d_out"] == 2
