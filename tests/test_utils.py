r"""Tests for the lampe.utils module."""

import math
import pytest
import torch

from lampe.utils import *
from torch import rand, arange


def test_bisection():
    f = torch.cos
    a = 2.0 + torch.rand(16, 1)
    b = torch.rand(8)

    x = bisection(f, a, b, n=21)

    assert x.shape == (16, 8)
    assert torch.allclose(x, torch.tensor(math.pi / 2), atol=1e-5)
    assert torch.allclose(f(x), torch.tensor(0.0), atol=1e-5)


def test_broadcast():
    # Trivial
    a = rand(2, 3)
    (b,) = broadcast(a)

    assert a.shape == b.shape
    assert (a == b).all()

    # Standard
    a, b, c = rand(1).squeeze(), rand(2), rand(3, 1)
    d, e, f = broadcast(a, b, c)

    assert d.shape == e.shape == f.shape == (3, 2)
    assert (a == d).all() and (b == e).all() and (c == f).all()

    # Invalid
    with pytest.raises(RuntimeError):
        a, b = rand(2), rand(3)
        d, e = broadcast(a, b)

    # Ignore last dimension
    a, b = rand(2), rand(3, 4)
    c, d = broadcast(a, b, ignore=1)

    assert c.shape == (3, 2) and d.shape == (3, 4)
    assert (a == c).all() and (b == d).all()

    # Ignore mixed dimensions
    a, b = rand(2, 3), rand(3, 4)
    c, d = broadcast(a, b, ignore=[0, 1])

    assert c.shape == (2, 3) and d.shape == (2, 3, 4)
    assert (a == c).all() and (b == d).all()


def test_deepto():
    # Tensor
    a = arange(5)
    b = deepto(a, dtype=a.dtype)
    c = deepto(a, dtype=torch.float)

    assert a is b and a is not c
    assert a.shape == b.shape == c.shape
    assert c.dtype is torch.float

    # String
    a = 'text'
    b = deepto(a, dtype=torch.float)

    assert a is b

    # Tuple
    a = (arange(5), 'text', rand(3), None)
    b = deepto(a, dtype=torch.float)

    assert a is not b
    assert type(a) == type(b)
    assert len(a) == len(b)
    assert b[0].dtype == torch.float
    assert b[1] == 'text'
    assert b[2].dtype == torch.float

    # List
    c = list(a)
    b = deepto(c, dtype=torch.long)

    assert c is b
    assert c[0].dtype == torch.long
    assert c[1] == 'text'
    assert c[2].dtype == torch.long

    # Dict
    c = dict(enumerate(a))
    b = deepto(c, dtype=torch.float)

    assert c is b
    assert c[0].dtype == torch.float
    assert c[1] == 'text'
    assert c[2].dtype == torch.float

    # Object
    class A:
        def __init__(self):
            self.text = 'text'
            self.loc = rand(5)
            self.scale = rand(5)

    a = A()
    b = deepto(a, dtype=torch.double)

    assert a is b
    assert a.text == 'text'
    assert a.loc.dtype == torch.double
    assert a.scale.dtype == torch.double


def test_gauss_legendre():
    # Polynomial
    f = lambda x: x**5 - x**2
    F = lambda x: x**6 / 6 - x**3 / 3
    a, b = 5 * rand(2, 64)

    area = gauss_legendre(f, a, b, n=3)

    assert torch.allclose(F(b) - F(a), area, atol=1e-5, rtol=1e-3)

    # Gradients
    grad_a, grad_b = torch.autograd.functional.jacobian(
        lambda a, b: gauss_legendre(f, a, b).sum(),
        (a, b),
    )

    assert torch.allclose(-f(a), grad_a)
    assert torch.allclose(f(b), grad_b)


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

    x, y = gridapply(f, bins=10, bounds=(lower, upper))

    assert x.shape == (10, 10, 10, 3)
    assert (lower <= x).all() and (x <= upper).all()
    assert y.shape == (10, 10, 10)
    assert torch.allclose(f(x), y)
