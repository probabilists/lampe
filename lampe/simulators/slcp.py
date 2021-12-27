r"""Simple Likelihood Complex Posterior (SLCP) simulator"""

import torch
import torch.nn as nn

from torch import Tensor, BoolTensor

from . import Simulator
from .priors import Distribution, Independent, JointUniform, MultivariateNormal


class SLCP(Simulator):
    r"""Simple Likelihood Complex Posterior"""

    def __init__(self, limit: float = 3.):
        super().__init__()

        self.register_buffer('low', torch.full((5,), -limit))
        self.register_buffer('high', torch.full((5,), limit))

    def marginal_prior(self, mask: BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return JointUniform(self.low[mask], self.high[mask])

    def likelihood(self, theta: Tensor, eps: float = 1e-8) -> Distribution:
        r""" p(x | theta) """

        # Mean
        mu = theta[..., :2]

        # Covariance
        s1 = theta[..., 2] ** 2 + eps
        s2 = theta[..., 3] ** 2 + eps
        rho = theta[..., 4].tanh()

        cov = stack2d([
            [      s1 ** 2, rho * s1 * s2],
            [rho * s1 * s2,       s2 ** 2],
        ])

        # Repeat
        mu = mu.unsqueeze(-2).repeat_interleave(4, -2)
        cov = cov.unsqueeze(-3).repeat_interleave(4, -3)

        # Normal
        normal = MultivariateNormal(mu, cov)

        return Independent(normal, 1)


def stack2d(matrix: list[list[Tensor]]) -> Tensor:
    return torch.stack([
        torch.stack(row, dim=-1)
        for row in matrix
    ], dim=-2)
