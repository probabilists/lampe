r"""Tests for the lampe.nn module."""

import pytest
import torch

from lampe.nn import *
from torch import randn


def test_ResMLP():
    net = ResMLP(3, 5)

    # Non-batched
    x = randn(3)
    y = net(x)

    assert y.shape == (5,)
    assert y.requires_grad

    # Batched
    x = randn(256, 3)
    y = net(x)

    assert y.shape == (256, 5)
