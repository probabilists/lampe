r"""Neural ratio estimation (NRE) components.

The principle of neural ratio estimation is to train a classifier network
:math:`d_\phi(\theta, x)` to discriminate between pairs :math:`(\theta, x)` equally
sampled from the joint distribution :math:`p(\theta, x)` and the product of the
marginals :math:`p(\theta)p(x)`. Formally, the optimization problem is

.. math:: \arg\min_\phi
    \frac{1}{2} \mathbb{E}_{p(\theta, x)} \big[ \ell(d_\phi(\theta, x)) \big] +
    \frac{1}{2} \mathbb{E}_{p(\theta)p(x)} \big[ \ell(1 - d_\phi(\theta, x)) \big]

where :math:`\ell(p) = -\log p` is the negative log-likelihood. For this task, the
decision function modeling the Bayes optimal classifier is

.. math:: d(\theta, x) = \frac{p(\theta, x)}{p(\theta, x) + p(\theta) p(x)}

thereby defining the likelihood-to-evidence (LTE) ratio

.. math:: r(\theta, x)
    = \frac{d(\theta, x)}{1 - d(\theta, x)}
    = \frac{p(\theta, x)}{p(\theta) p(x)}
    = \frac{p(x | \theta)}{p(x)}
    = \frac{p(\theta | x)}{p(\theta)} .

To prevent numerical stability issues when :math:`d_\phi(\theta, x) \to 0`, the neural
network returns the logit of the class prediction :math:`\text{logit}(d_\phi(\theta, x))
= \log r_\phi(\theta, x)`.

References:
    | Approximating Likelihood Ratios with Calibrated Discriminative Classifiers (Cranmer et al., 2015)
    | https://arxiv.org/abs/1506.02169

    | Likelihood-free MCMC with Amortized Approximate Ratio Estimators (Hermans et al., 2019)
    | https://arxiv.org/abs/1903.04057
"""

__all__ = [
    "NRE",
    "NRELoss",
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable
from zuko.utils import broadcast

# isort: split
from ..nn import MLP


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) network.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        build: The network constructor (e.g. :class:`lampe.nn.ResMLP`). It takes the
            number of input and output features as positional arguments.
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.net = build(theta_dim + x_dim, 1, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta, x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class NRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for a NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
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

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (l1 + l0) / 2
