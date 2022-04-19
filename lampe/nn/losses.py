r"""Training losses and routines."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.distributions import Distribution
from typing import *


class NRELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a NRE classifier
    :math:`d_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = - \log p` is the negative log-likelihood.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x)`.
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
            torch.stack((x, x)),
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class AMNRELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a AMNRE classifier
    :math:`d_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i \odot b_i, x_i, b_i)) +
        \ell(1 - d_\phi(\theta_{i+1} \odot b_i, x_i, b_i))

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(self, estimator: nn.Module, mask_dist: Distribution):
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

        b = self.mask_dist.sample(theta.shape[:-1])
        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            torch.stack((x, x)),
            b,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class NPELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of a NPE normalizing flow
    :math:`p_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N -\log p_\phi(\theta_i | x_i) .

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x)`.
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


class AMNPELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l` of an AMNPE normalizing flow
    :math:`p_\phi`. Given a batch of :math:`N` pairs :math:`\{ (\theta_i, x_i) \}`,
    the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        -\log p_\phi(\theta_i \odot b_i + \theta_{i + 1} \odot (1 - b_i) | x_i, b_i)

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(self, estimator: nn.Module, mask_dist: Distribution):
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

        b = self.mask_dist.sample(theta.shape[:-1])
        theta_prime = torch.roll(theta, 1, dims=0)
        theta = torch.where(b, theta, theta_prime)

        log_prob = self.estimator(theta, x, b)

        return -log_prob.mean()
