r"""Neural posterior estimation (NPE) components.

The principle of neural posterior estimation is to train a parametric conditional
distribution :math:`p_\phi(\theta | x)` to approximate the posterior distribution
:math:`p(\theta | x)`. The optimization problem is to minimize the expected
Kullback-Leibler (KL) divergence between the two distributions for all observations
:math:`x \sim p(x)`, that is,

.. math::
    \arg\min_\phi & ~ \mathbb{E}_{p(x)}
        \Big[ \text{KL} \big( p(\theta|x) \parallel p_\phi(\theta | x) \big) \Big] \\
    = \arg\min_\phi & ~ \mathbb{E}_{p(x)} \, \mathbb{E}_{p(\theta | x)}
        \left[ \log \frac{p(\theta | x)}{p_\phi(\theta | x)} \right] \\
    = \arg\min_\phi & ~ \mathbb{E}_{p(\theta, x)}
        \big[ -\log p_\phi(\theta | x) \big] .

Normalizing flows are typically used for :math:`p_\phi(\theta | x)` as they are
differentiable parametric distributions enabling gradient-based optimization techniques.

Wikipedia:
    https://wikipedia.org/wiki/Kullback-Leibler_divergence
"""

__all__ = [
    'NPE',
    'NPELoss',
]

import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.flows import MAF, Flow
from zuko.utils import broadcast


class NPE(nn.Module):
    r"""Creates a neural posterior estimation (NPE) normalizing flow.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        build: The flow constructor (e.g. :class:`zuko.flows.spline.NSF`). It takes
            the number of sample and context features as positional arguments.
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], Flow] = MAF,
        **kwargs,
    ):
        super().__init__()

        self.flow = build(theta_dim, x_dim, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.flow(x).log_prob(theta)


class NPELoss(nn.Module):
    r"""Creates a module that calculates the negative log-likelihood (NLL) loss for a
    NPE density estimator.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N -\log p_\phi(\theta_i | x_i) .

    Arguments:
        estimator: A log-density estimator :math:`\log p_\phi(\theta | x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x)

        return -log_p.mean()
