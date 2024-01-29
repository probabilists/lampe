r"""Balanced neural ratio estimation (BNRE) components.

References:
    | Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation (Delaunoy et al., 2022)
    | https://arxiv.org/abs/2208.13624
"""

__all__ = [
    'BNRELoss',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class BNRELoss(nn.Module):
    r"""Creates a module that calculates the balanced cross-entropy loss for NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{2N} \sum_{i = 1}^N
            \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\theta_i, x_i) + d_\phi(\theta_{i+1}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        lmbda: The weight :math:`\lambda` controlling the strength of the balancing
            condition.
    """

    def __init__(self, estimator: nn.Module, lmbda: float = 100.0):
        super().__init__()

        self.estimator = estimator
        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()

        return (l1 + l0) / 2 + self.lmbda * lb
