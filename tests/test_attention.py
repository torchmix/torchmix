import pytest
import torch

from torchmix import Attention, WindowAttention


@pytest.mark.slow
def test_self_attention():
    x = torch.randn(2, 196, 128)
    module = Attention(
        dim=128,
        num_heads=8,
        head_dim=64,
    )
    assert module(x).shape == x.shape


@pytest.mark.slow
def test_window_attention():
    x = torch.randn(2, 3136, 96)
    module = WindowAttention(
        dim=96,
        num_heads=8,
        head_dim=64,
        window_size=7,
    )
    assert module(x).shape == x.shape
