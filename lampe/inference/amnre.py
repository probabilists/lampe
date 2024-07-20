r"""Arbitrary marginal neural ratio estimation (AMNRE) components.

The principle of AMNRE is to introduce, as input to the classifier, a binary mask
:math:`b \in \{0, 1\}^D` indicating a subset of parameters :math:`\theta_b = (\theta_i:
b_i = 1)` of interest. Intuitively, this allows the classifier to distinguish subspaces
and to learn a different ratio for each of them. Formally, the classifier network takes
the form :math:`d_\phi(\theta_b, x, b)` and the optimization problem becomes

.. math:: \arg\min_\phi
    \frac{1}{2} \mathbb{E}_{p(\theta, x) P(b)} \big[ \ell(d_\phi(\theta_b, x, b)) \big] +
    \frac{1}{2} \mathbb{E}_{p(\theta)p(x) P(b)} \big[ \ell(1 - d_\phi(\theta_b, x, b)) \big],

where :math:`P(b)` is a binary mask distribution. In this context, the Bayes optimal
classifier is

.. math:: d(\theta_b, x, b)
    = \frac{p(\theta_b, x)}{p(\theta_b, x) + p(\theta_b) p(x)}
    = \frac{r(\theta_b, x)}{1 + r(\theta_b, x)} .

Therefore, a classifier network trained for AMNRE gives access to an estimator
:math:`\log r_\phi(\theta_b, x, b)` of all marginal LTE log-ratios :math:`\log
r(\theta_b, x)`.

References:
    | Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference (Rozet et al., 2021)
    | https://arxiv.org/abs/2110.00449
"""

__all__ = [
    'AMNRE',
    'AMNRELoss',
]

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import BoolTensor, Tensor
from torch.distributions import Distribution
from zuko.utils import broadcast

# isort: split
from .nre import NRE


class AMNRE(NRE):
    r"""Creates an arbitrary marginal neural ratio estimation (AMNRE) network.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        args: Positional arguments passed to :class:`lampe.inference.nre.NRE`.
        kwargs: Keyword arguments passed to :class:`lampe.inference.nre.NRE`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim, x_dim + theta_dim, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`, or
                a subset :math:`\theta_b`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta_b, x, b)`, with shape :math:`(*,)`.
        """

        if theta.shape[-1] < b.shape[-1]:
            theta, b = broadcast(theta, b, ignore=1)
            theta = theta.new_zeros(b.shape).masked_scatter(b, theta)

        theta, x, b = broadcast(theta * b, x, b * 2.0 - 1.0, ignore=1)

        return self.net(torch.cat((theta, x, b), dim=-1)).squeeze(-1)


class AMNRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for an AMNRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i \odot b_i, x_i, b_i)) +
        \ell(1 - d_\phi(\theta_{i+1} \odot b_i, x_i, b_i))

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
    ):
        super().__init__()

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        b = self.mask_dist.sample(theta.shape[:-1])

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
            b,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (l1 + l0) / 2
