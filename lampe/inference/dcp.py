r"""Inference components such as estimators, training losses and MCMC samplers."""

__all__ = [
    'DCPNRELoss',
    'DCPNPELoss',
]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsort

from torch import Tensor, Size
from typing import *


class STEhardtanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class DCPNRELoss(nn.Module):
    r"""Creates a module that calculates cross-entropy loss and the
    calibration/conservativeness loss for an NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{2N} \sum_{i = 1}^N
            \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i)) \\
          & + \lambda 1/M \sum_{j=1}^M | \text{ECP}(1 - \alpha_j) - (1 - \alpha_j)|

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and
    :math:`\text{ECP}(1 - \alpha_j)` is the Expected Coverage Probability at
    credibility level :math:`1 - \alpha_j`.

    References:
        | Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability (Falkiewicz et al., 2023)
        | https://arxiv.org/abs/2310.13402

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        prior: Prior distribution :math:`p(\theta)`.
        proposal: If given, proposal distribution module for Importance Sampling :math:`I(\theta)`, otherwise prior is used for IS.
        lmbda: The weight :math:`\lambda` controlling the strength of the regularizer.
        n_samples: Number of samples in MC estimate of rank statistic
        calibration: Boolean flag of calibration objective (default: False)
        sort_kwargs: Arguments of differentiable sorting algorithm, see [doc](https://github.com/teddykoker/torchsort#usage) (default: None)
    """

    def __init__(
        self,
        estimator: nn.Module,
        prior: torch.distributions.Distribution,
        proposal: torch.distributions.Distribution = None,
        lmbda: float = 1.0,
        n_samples: int = 16,
        calibration: bool = False,
        sort_kwargs: dict = None,
    ):
        super().__init__()

        self.estimator = estimator
        self.prior = prior
        if proposal is None:
            self.proposal = prior
        else:
            self.proposal = proposal
        self.lmbda = lmbda
        self.n_samples = n_samples
        if calibration:
            self.activation = lambda input: torch.abs(input)
        else:
            self.activation = torch.nn.ReLU()
        if sort_kwargs is None:
            self.sort_kwargs = dict()
        else:
            self.sort_kwargs = sort_kwargs

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """
        theta_prime = torch.roll(theta, 1, dims=0)
        theta_is = self.proposal.sample(
            (
                self.n_samples,
                theta.shape[0],
            )
        )

        log_r_all = self.estimator(
            theta_all := torch.cat((torch.stack((theta_prime, theta)), theta_is)),
            x,
        )
        log_r_all[0].neg_()

        logq_nominal_and_is = log_r_all[1:] + self.prior.log_prob(theta_all[1:])
        return -F.logsigmoid(log_r_all[:2]).mean() + self.lmbda * self.regularizer(
            logq_nominal_and_is, theta_is
        )

    def get_cdfs(self, ranks):
        alpha = torchsort.soft_sort(ranks.unsqueeze(0), **self.sort_kwargs).squeeze()
        return (
            torch.linspace(0.0, 1.0, len(alpha) + 1, device=alpha.device)[1:],
            alpha,
        )

    def get_rank_statistics(
        self,
        logq: Tensor,
        is_samples: Tensor,
    ):
        q = logq.exp()
        is_log_weights = logq[1:, :] - self.proposal.log_prob(is_samples)
        return (
            (is_log_weights - is_log_weights.logsumexp(dim=0, keepdims=True)).exp()
            * STEhardtanh.apply(q[0, :].unsqueeze(0) - q[1:, :])
        ).sum(dim=0)

    def regularizer(self, logq, is_samples) -> Tensor:
        ranks = self.get_rank_statistics(logq, is_samples)
        target_cdf, ecdf = self.get_cdfs(ranks)
        return self.activation(target_cdf - ecdf).mean()


class DCPNPELoss(nn.Module):
    r"""Creates a module that calculates the negative log-likelihood loss and
    the calibration/conservativeness loss for a NPE normalizing flow.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{N} \sum_{i = 1}^N -\log p_\phi(\theta_i | x_i) \\
          & + \lambda 1/M \sum_{j=1}^M | \text{ECP}(1 - \alpha_j) - (1 - \alpha_j)|

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and
    :math:`\text{ECP}(1 - \alpha_j)` is the Expected Coverage Probability at
    credibility level :math:`1 - \alpha_j`.

    References:
        | Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability (Falkiewicz et al., 2023)
        | https://arxiv.org/abs/2310.13402

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x)`.
        proposal: If given, proposal distribution module for Importance Sampling :math:`I(\theta)`, otherwise regularizer is estimated by directly sampling from the model (default: None).
        lmbda: The weight :math:`\lambda` controlling the strength of the regularizer.
        n_samples: Number of samples in MC estimate of rank statistic
        calibration: Boolean flag of calibration objective (default: False)
        sort_kwargs: Arguments of differentiable sorting algorithm, see [doc](https://github.com/teddykoker/torchsort#usage) (default: None)
    """

    def __init__(
        self,
        estimator: nn.Module,
        proposal: torch.distributions.Distribution = None,
        lmbda: float = 1.0,
        n_samples: int = 16,
        calibration: bool = False,
        sort_kwargs: dict = None,
    ):
        super().__init__()

        self.estimator = estimator
        self.proposal = proposal
        self.lmbda = lmbda
        self.n_samples = n_samples
        if calibration:
            self.activation = lambda input: torch.abs(input)
        else:
            self.activation = torch.nn.ReLU()
        if sort_kwargs is None:
            self.sort_kwargs = dict()
        else:
            self.sort_kwargs = sort_kwargs
        if self.proposal is None:
            self.forward = self._forward
            self.get_rank_statistics = self._get_rank_statistics
        else:
            self.forward = self._forward_is
            self.get_rank_statistics = self._get_rank_statistics_is

    def rsample_and_log_prob(
        self, x: Tensor, shape: Size = ()
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            A tuple containing samples from the model
            :math:`\log p_\phi(\theta | x)`, with shape :math:`(*, D, *shape)`
            and their log-density :math:`\log p_\phi(\theta | x)`,
            with shape :math:`(*, *shape)`.
        """

        return tuple(
            map(
                lambda t: t.movedim(1, 0),
                self.estimator.flow(x).rsample_and_log_prob(shape),
            )
        )

    def _forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """
        log_p = self.estimator(theta, x)
        lr = self.regularizer(x, log_p)

        return -log_p.mean() + self.lmbda * lr

    def _forward_is(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """
        theta_is = self.proposal.sample(
            (
                self.n_samples,
                theta.shape[0],
            )
        )
        log_p = self.estimator(torch.cat((theta.unsqueeze(0), theta_is)), x)
        lr = self.regularizer(x, log_p, theta_is)

        return -log_p[0].mean() + self.lmbda * lr

    def get_cdfs(self, ranks):
        alpha = torchsort.soft_sort(ranks.unsqueeze(0), **self.sort_kwargs).squeeze()
        return (
            torch.linspace(0.0, 1.0, len(alpha) + 1, device=alpha.device)[1:],
            alpha,
        )

    def _get_rank_statistics(self, x, logq, *args):
        q = torch.cat(
            [
                logq.unsqueeze(-1),
                self.rsample_and_log_prob(
                    x,
                    (self.n_samples,),
                )[1],
            ],
            dim=1,
        ).exp()
        return STEhardtanh.apply(q[:, 0].unsqueeze(1) - q[:, 1:]).mean(dim=1)

    def _get_rank_statistics_is(self, x, logq, is_samples):
        q = logq.exp()
        is_log_weights = logq[1:, :] - self.proposal.log_prob(is_samples)
        return (
            (is_log_weights - is_log_weights.logsumexp(dim=0, keepdims=True)).exp()
            * STEhardtanh.apply(q[0, :].unsqueeze(0) - q[1:, :])
        ).sum(dim=0)

    def regularizer(self, x: Tensor, logq, is_samples=None) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar regularizer term :math:`r`.
        """
        ranks = self.get_rank_statistics(x, logq, is_samples)
        target_cdf, ecdf = self.get_cdfs(ranks)
        return self.activation(target_cdf - ecdf).mean()
