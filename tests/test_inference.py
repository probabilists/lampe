r"""Tests for the lampe.inference module."""

import pytest
import torch

from lampe.inference import *
from lampe.masks import BernoulliMask
from torch import randn


def test_NRE():
    estimator = NRE(3, 5)

    # Non-batched
    theta, x = randn(3), randn(5)
    log_r = estimator(theta, x)

    assert log_r.shape == ()
    assert log_r.requires_grad

    # Batched
    theta, x = randn(256, 3), randn(256, 5)
    log_r = estimator(theta, x)

    assert log_r.shape == (256,)

    # Mixed
    theta, x = randn(256, 3), randn(5)
    log_r = estimator(theta, x)

    assert log_r.shape == (256,)


def test_NRELoss():
    estimator = NRE(3, 5)
    loss = NRELoss(estimator)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_BNRELoss():
    estimator = NRE(3, 5)
    loss = BNRELoss(estimator)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_CNRELoss():
    estimator = NRE(3, 5)
    loss = CNRELoss(estimator, 2, 1.0)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_BinaryBalancedCNRELoss():
    estimator = NRE(3, 5)
    loss = BinaryBalancedCNRELoss(estimator, 2, 1.0, 1.0)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_AMNRE():
    estimator = AMNRE(3, 5)

    # Non-batched
    theta, x, b = randn(3), randn(5), randn(3) < 0
    log_r = estimator(theta, x, b)

    assert log_r.shape == ()
    assert log_r.requires_grad

    # Batched
    theta, x, b = randn(256, 3), randn(256, 5), randn(256, 3) < 0
    log_r = estimator(theta, x, b)

    assert log_r.shape == (256,)

    grad = torch.autograd.functional.jacobian(lambda theta: estimator(theta, x, b).sum(), theta)

    assert (grad[~b] == 0).all()

    # Mixed
    theta, x, b = randn(256, 3), randn(5), randn(2, 1, 3)
    log_r = estimator(theta, x, b)

    assert log_r.shape == (2, 256)

    # Subset
    theta, x, b = randn(256, 3), randn(256, 5), torch.tensor([True, False, True])

    log_r1 = estimator(theta, x, b)
    log_r2 = estimator(theta[..., b], x, b)

    assert torch.allclose(log_r1, log_r2)


def test_AMNRELoss():
    estimator = AMNRE(3, 5)
    mask_dist = BernoulliMask(0.5 * torch.ones(3))
    loss = AMNRELoss(estimator, mask_dist)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_NPE():
    estimator = NPE(3, 5)

    # Non-batched
    theta, x = randn(3), randn(5)
    log_p = estimator(theta, x)

    assert log_p.shape == ()
    assert log_p.requires_grad

    # Batched
    theta, x = randn(256, 3), randn(256, 5)
    log_p = estimator(theta, x)

    assert log_p.shape == (256,)

    # Mixed
    theta, x = randn(256, 3), randn(5)
    log_p = estimator(theta, x)

    assert log_p.shape == (256,)

    # Sample
    x = randn(32, 5)
    theta = estimator.sample(x, (8,))

    assert theta.shape == (8, 32, 3)


def test_NPELoss():
    estimator = NPE(3, 5)
    loss = NPELoss(estimator)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_AMNPE():
    estimator = AMNPE(3, 5)

    # Non-batched
    theta, x, b = randn(3), randn(5), randn(3) < 0
    log_p = estimator(theta, x, b)

    assert log_p.shape == ()
    assert log_p.requires_grad

    # Batched
    theta, x, b = randn(256, 3), randn(256, 5), randn(256, 3) < 0
    log_p = estimator(theta, x, b)

    assert log_p.shape == (256,)

    # Mixed
    theta, x, b = randn(256, 3), randn(5), randn(2, 1, 3)
    log_r = estimator(theta, x, b)

    assert log_r.shape == (2, 256)

    # Sample
    x, b = randn(32, 5), torch.tensor([True, False, True])
    theta = estimator.sample(x, b, (8,))

    assert theta.shape == (8, 32, 2)


def test_AMNPELoss():
    estimator = AMNPE(3, 5)
    mask_dist = BernoulliMask(0.5 * torch.ones(3))
    loss = AMNRELoss(estimator, mask_dist)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_NSE():
    estimator = NSE(3, 5)

    # Non-batched
    theta, x, t = randn(3), randn(5), torch.tensor(0.5)
    score = estimator(theta, x, t)

    assert score.shape == (3,)
    assert score.requires_grad

    # Batched
    theta, x, t = randn(256, 3), randn(256, 5), randn(256)
    score = estimator(theta, x, t)

    assert score.shape == (256, 3)

    # Mixed
    theta, x, t = randn(256, 3), randn(5), randn(1)
    score = estimator(theta, x, t)

    assert score.shape == (256, 3)

    # Sample
    x = randn(32, 5)
    theta = estimator.flow(x).sample((8,))

    assert theta.shape == (8, 32, 3)

    # Log-density
    log_p = estimator.flow(x).log_prob(theta)

    assert log_p.shape == (8, 32)


def test_NSELoss():
    estimator = NSE(3, 5)
    loss = NSELoss(estimator)

    theta, x = randn(256, 3), randn(256, 5)

    l = loss(theta, x)

    assert l.shape == ()
    assert l.requires_grad


def test_MetropolisHastings():
    log_f = lambda x: -(x**2).sum(dim=-1) / 2
    f = lambda x: torch.exp(log_f(x))

    # f
    sampler = MetropolisHastings(randn(32, 3), f=f, sigma=0.5)

    it = iter(sampler)
    x1 = next(it)
    x2 = next(it)

    assert x1.shape == x2.shape == (32, 3)
    assert not (x1 == x2).all()

    # log_f
    sampler = MetropolisHastings(randn(32, 3), log_f=log_f, sigma=0.5)

    xs = list(sampler(256, burn=128, step=8))

    assert len(xs) == (256 - 128) // 8
    assert not (xs[0] == xs[1]).all()
