r"""Simple Likelihood Complex Posterior (SLCP)

SLCP [1] is a toy simulator where theta parametrizes a 2-d multivariate Gaussian
from which 4 points are independently drawn and stacked as a single observation x.

It is a non-trivial parameter inference benchmark that allows to retrieve
the ground-truth posterior through MCMC sampling of its tractable likelihood.

References:
    [1] Sequential Neural Likelihood: Fast Likelihood-free Inference with Autoregressive Flows
    (Papamakarios et al., 2019)
    https://arxiv.org/abs/1805.07226

Shapes:
    theta: (5,)
    x: (4, 2)
"""

import torch
import torch.nn as nn

from torch import Tensor, BoolTensor
from typing import *

from . import Simulator
from ..priors import Distribution, JointUniform, MultivariateNormal


labels = [f'$\\theta_{{{i + 1}}}$' for i in range(5)]


lower = torch.full((5,), -3.)
upper = torch.full((5,), 3.)


def slcp_prior(mask: BoolTensor = None) -> Distribution:
    r""" p(theta) """

    if mask is None:
        mask = ...

    return JointUniform(lower[mask], upper[mask])


class SLCP(Simulator):
    r"""Simple Likelihood Complex Posterior (SLCP) simulator"""

    def likelihood(self, theta: Tensor, eps: float = 1e-8) -> Distribution:
        r""" p(x | theta) """

        # Mean
        mu = theta[..., :2]

        # Covariance
        s1 = theta[..., 2] ** 2 + eps
        s2 = theta[..., 3] ** 2 + eps
        rho = theta[..., 4].tanh()

        cov = torch.stack([
            s1 ** 2, rho * s1 * s2,
            rho * s1 * s2, s2 ** 2,
        ], dim=-1)

        cov = cov.view(cov.shape[:-1] + (2, 2))

        # Repeat 4 times
        mu = mu.unsqueeze(-2).repeat_interleave(4, -2)
        cov = cov.unsqueeze(-3).repeat_interleave(4, -3)

        # Normal
        return MultivariateNormal(mu, cov)

    def __call__(self, theta: Tensor) -> Tensor:
        r""" x ~ p(x | theta) """

        return self.likelihood(theta).sample()

    def log_prob(self, theta: Tensor, x: Tensor) -> Tensor:
        r""" log p(x | theta) """

        return self.likelihood(theta).log_prob(x).sum(dim=-1)
