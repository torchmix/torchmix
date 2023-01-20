import torch

from torchmix import Add, Attach, ClassEmbedding, nn


def test_add():
    module = Add(
        nn.Linear(10, 20),
        nn.Linear(10, 20),
        nn.Sequential(
            nn.Linear(10, 15),
            nn.GELU(),
            nn.Linear(15, 20),
        ),
        nn.Linear(10, 20),
    )
    assert module(torch.randn(32, 10)).shape == (32, 20)


def test_attach():
    module = Attach(
        nn.Linear(4, 10),
        ClassEmbedding(10),
        ClassEmbedding(10),
        ClassEmbedding(10),
        ClassEmbedding(10),
    )

    assert module(torch.randn(2, 3, 4)).shape == (2, 3 + 4, 10)
