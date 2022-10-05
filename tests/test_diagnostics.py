r"""Tests for the lampe.diagnostics module."""

import pytest
import torch

from lampe.diagnostics import *
from zuko.distributions import Independent, Normal, Truncated


def test_expected_coverage_mc():
    posterior = lambda x: Independent(Normal(0, 1 + x**2), 1)
    x = torch.randn(1024, 2)
    theta = posterior(x).sample()
    pairs = list(zip(theta, x))

    # Exact
    estimator = posterior
    levels, coverages = expected_coverage_mc(estimator, pairs, n=1024)

    assert torch.allclose(levels, coverages, atol=1e-1)

    # Conservative
    estimator = lambda x: Independent(Normal(0, 2 + x**2), 1)
    levels, coverages = expected_coverage_mc(estimator, pairs, n=1024)

    assert (coverages > levels).float().mean() > 0.9

    # Overconfident
    estimator = lambda x: Independent(Normal(0, 0.5 + x**2), 1)
    levels, coverages = expected_coverage_mc(estimator, pairs, n=1024)

    assert (coverages < levels).float().mean() > 0.9


def test_expected_coverage_ni():
    posterior = lambda x: Independent(Truncated(Normal(0, 1 + x**2), -3, 3), 1)
    x = torch.randn(1024, 2)
    theta = posterior(x).sample()
    pairs = list(zip(theta, x))
    domain = (-3 * torch.ones(2), 3 * torch.ones(2))

    # Exact
    estimator = lambda theta, x: posterior(x).log_prob(theta)
    levels, coverages = expected_coverage_ni(estimator, pairs, domain, bins=128)

    assert torch.allclose(levels, coverages, atol=1e-1)

    # Conservative
    estimator = lambda theta, x: Independent(
        Truncated(Normal(0, 2 + x**2), -3, 3), 1
    ).log_prob(theta)
    levels, coverages = expected_coverage_ni(estimator, pairs, domain, bins=128)

    assert (coverages > levels).float().mean() > 0.9

    # Overconfident
    estimator = lambda theta, x: Independent(
        Truncated(Normal(0, 0.5 + x**2), -3, 3), 1
    ).log_prob(theta)
    levels, coverages = expected_coverage_ni(estimator, pairs, domain, bins=128)

    assert (coverages < levels).float().mean() > 0.9
