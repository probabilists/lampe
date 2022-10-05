r"""Tests for the lampe.utils module."""

import math
import pytest
import torch

from lampe.utils import *
from torch import rand


def test_GDStep():
    layer = torch.nn.Linear(3, 3)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
    step = GDStep(optimizer)

    # Normal step
    before = layer.weight.detach().clone()

    x = rand(3)
    y = layer(x)

    loss = (y - x).square().sum()
    loss = step(loss)

    after = layer.weight.detach().clone()

    assert not loss.requires_grad
    assert not torch.allclose(before, after)

    # Detached loss
    with pytest.raises(RuntimeError):
        step(loss)

    # Non-scalar loss
    with pytest.raises(RuntimeError):
        step(layer(x))

    # Non-finite loss
    before = layer.weight.detach().clone()

    x = rand(3) * float('inf')
    y = layer(x)

    loss = (y - x).square().sum()
    loss = step(loss)

    after = layer.weight.detach().clone()

    assert torch.allclose(before, after)

    # Non-finite grad with clip
    step.clip = 1

    before = layer.weight.detach().clone()

    x = rand(3)
    y = layer(x)

    loss = (y.sum() * 0).sqrt()
    loss = step(loss)

    after = layer.weight.detach().clone()

    assert torch.allclose(before, after)

    # Non-finite grad without clip
    step.clip = None

    x = rand(3)
    y = layer(x)

    loss = (y.sum() * 0).sqrt()
    loss = step(loss)

    assert torch.isnan(layer.weight).all()


def test_gridapply():
    f = lambda x: x.square().sum(dim=-1)
    lower, upper = torch.zeros(3), torch.ones(3)

    x, y = gridapply(f, (lower, upper), bins=8)

    assert x.shape == (8, 8, 8, 3)
    assert (lower <= x).all() and (x <= upper).all()
    assert y.shape == (8, 8, 8)
    assert torch.allclose(f(x), y)
