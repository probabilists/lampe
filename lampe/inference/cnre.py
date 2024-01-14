r"""Contrastive neural ratio estimation (CNRE) components.

The principle of contrastive neural ratio estimation (CNRE) is to predict whether a set
:math:`\Theta = \{\theta^1, \dots, \theta^K\}` contains or not the parameters that
originated an observation :math:`x`. The elements of :math:`\Theta` are drawn
independently from the prior :math:`p(\theta)` and the element :math:`\theta^k` that
originates the observation :math:`x \sim p(x | \theta^k)` is chosen uniformly within
:math:`\Theta`, such that

.. math:: p(\Theta, x)
    & = p(\Theta) \, p(x | \Theta) \\
    & = p(\Theta) \frac{1}{K} \sum_{k = 1}^K p(x | \theta^k) \\
    & = p(\Theta) \, p(x) \frac{1}{K} \sum_{k = 1}^K r(\theta^k, x)

where :math:`r(\theta, x)` is the likelihood-to-evidence (LTE) ratio. The task is to
discriminate between pairs :math:`(\Theta, x)` for which :math:`\Theta` either does or
does not contain the nominal parameters of :math:`x`, similar to the original NRE
optimization problem. For this task, the decision function modeling the Bayes optimal
classifier is

.. math:: d(\Theta, x)
    = \frac{p(\Theta, x)}{p(\Theta, x) + \frac{1}{\gamma} p(\Theta) p(x)}
    = \frac{\sum_{k = 1}^K r(\theta^k, x)}{\frac{K}{\gamma} + \sum_{k = 1}^K r(\theta^k, x)} \, ,

where :math:`\gamma \in \mathbb{R}^+` are the odds of :math:`\Theta` containing to not
containing the nominal parameters. Consequently, a classifier :math:`d_\phi(\Theta, x)`
can be equivalently replaced and trained as a composition of ratios
:math:`r_\phi(\theta^k, x)`.

Note:
    The quantity :math:`d_\phi(\Theta, x)` corresponds to :math:`q_\phi(y \neq 0 |
    \Theta, x)` or :math:`1 - q_\phi(y = 0 | \Theta, x)` in the notations of Miller et
    al. (2022).

References:
    | Contrastive Neural Ratio Estimation (Miller et al., 2022)
    | https://arxiv.org/abs/2210.06170
"""

__all__ = [
    'CNRELoss',
    'BCNRELoss',
]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import *


class CNRELoss(nn.Module):
    r"""Creates a module that calculates the contrastive cross-entropy loss for a NRE
    network.

    Given a batch of :math:`N \geq 2K` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \frac{\gamma}{\gamma + 1} \ell(d_\phi(\Theta_i, x_i))
        + \frac{1}{\gamma + 1} \ell(1 - d_\phi(\Theta_{i+K}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and :math:`\Theta_i =
    \{\theta_i, \dots, \theta_{i+K-1}\}`.

    References:
        | Contrastive Neural Ratio Estimation (Miller et al., 2022)
        | https://arxiv.org/abs/2210.06170

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        cardinality: The cardinality :math:`K` of :math:`\Theta`.
        gamma: The odds ratio :math:`\gamma`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        cardinality: int = 2,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.estimator = estimator
        self.cardinality = cardinality
        self.gamma = gamma

    def logits(self, theta: Tensor, x: Tensor) -> Tensor:
        theta = torch.cat((theta, theta[: self.cardinality - 1]), dim=0)
        theta = theta.unfold(0, self.cardinality, 1)
        theta = theta.movedim(-1, 0)

        theta_prime = torch.roll(theta, self.cardinality, dims=1)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        shift = math.log(self.gamma / self.cardinality)

        log_r = torch.logsumexp(log_r, dim=0) + shift
        log_r_prime = torch.logsumexp(log_r_prime, dim=0) + shift

        return log_r, log_r_prime

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_r, log_r_prime = self.logits(theta, x)

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (self.gamma * l1 + l0) / (self.gamma + 1)


class BCNRELoss(CNRELoss):
    r"""Creates a module that calculates the balanced contrastive cross-entropy loss for a NRE network.

    Given a batch of :math:`N \geq 2K` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{N} \sum_{i = 1}^N
            \frac{\gamma}{\gamma + 1} \ell(d_\phi(\Theta_i, x_i))
            + \frac{1}{\gamma + 1} \ell(1 - d_\phi(\Theta_{i+K}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\Theta_i, x_i) + d_\phi(\Theta_{i+K}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and :math:`\Theta_i =
    \{\theta_i, \dots, \theta_{i+K-1}\}`.

    References:
        | Balancing Simulation-based Inference for Conservative Posteriors (Delaunoy et al., 2023)
        | https://arxiv.org/abs/2304.10978

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        cardinality: The cardinality :math:`K` of :math:`\Theta`.
        gamma: The odds ratio :math:`\gamma`.
        lmbda: The weight :math:`\lambda` controlling the strength of the balancing
            condition.
    """

    def __init__(
        self,
        estimator: nn.Module,
        cardinality: int = 2,
        gamma: float = 1.0,
        lmbda: float = 100.0,
    ):
        super().__init__(estimator, cardinality, gamma)

        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_r, log_r_prime = self.logits(theta, x)

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()

        return (self.gamma * l1 + l0) / (self.gamma + 1) + self.lmbda * lb
