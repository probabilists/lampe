r"""Diagnostics and reliability assessment."""

import torch

from torch import Tensor, Size
from torch.distributions import Distribution
from tqdm import tqdm
from typing import *

from .utils import gridapply


def expected_coverage_mc(
    posterior: Callable[[Tensor], Distribution],
    pairs: Iterable[Tuple[Tensor, Tensor]],
    n: int = 1024,
    device: str = None,
) -> Tuple[Tensor, Tensor]:
    r"""Estimates by Monte Carlo (MC) the expected coverages of a posterior estimator
    :math:`q(\theta | x)` over pairs :math:`(\theta^*, x^*) \sim p(\theta, x)`.

    The expected coverage at credible level :math:`1 - \alpha` is the probability of
    :math:`\theta^*` to be included in the highest density region of total probability
    :math:`1 - \alpha` of the posterior :math:`q(\theta | x^*)`, denoted
    :math:`\Theta_{q(\theta | x^*)}(1 - \alpha)`. To get the coverages, the proportion
    :math:`r^*` of samples :math:`\theta \sim q(\theta | x^*)` having a higher estimated
    density than :math:`\theta^*` is computed for each pair :math:`(\theta^*, x^*)`.
    Formally,

    .. math:: r^* = \mathbb{E}_{q(\theta | x^*)}
        \Big[ \mathbb{I} \big[ q(\theta | x^*) \geq q(\theta^* | x^*) \big] \Big] .

    Then, the expected coverage at credible level :math:`1 - \alpha` is the probability
    of :math:`r^*` to be lower than :math:`1 - \alpha`,

    .. math:: P \big( \theta^* \in \Theta_{q(\theta | x^*)}(1 - \alpha) \big)
        = P(r^* \leq 1 - \alpha) .

    In practice, Monte Carlo estimations of :math:`r^*` are used.

    References:
        | Averting A Crisis In Simulation-Based Inference (Hermans et al., 2021)
        | https://arxiv.org/abs/2110.06581

    Arguments:
        posterior: A posterior estimator :math:`q(\theta | x)`.
        pairs: An iterable of pairs :math:`(\theta^*, x^*) \sim p(\theta, x)`.
        n: The number of samples to draw from the posterior for each pair.
        device: The device on which the computations are performed.

    Returns:
        A vector of increasing credible levels and their respective expected coverages.

    Example:
        >>> posterior = lampe.inference.NPE(3, 4)
        >>> testset = lampe.data.H5Dataset('test.h5')
        >>> levels, coverages = expected_coverage_mc(posterior.flow, testset)
    """

    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit='pair'):
            if device is not None:
                theta, x = theta.to(device), x.to(device)

            dist = posterior(x)
            samples = dist.sample((n,))
            mask = dist.log_prob(theta) < dist.log_prob(samples)
            rank = mask.sum() / mask.numel()

            ranks.append(rank)

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, ranks.new_tensor((0.0, 1.0))))

    return (
        torch.sort(ranks).values,
        torch.linspace(0, 1, len(ranks)),
    )


def expected_coverage_ni(
    log_p: Callable[[Tensor, Tensor], Tensor],
    pairs: Iterable[Tuple[Tensor, Tensor]],
    domain: Tuple[Tensor, Tensor],
    device: str = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    r"""Estimates by numerical integration (NI) the expected coverages of a posterior
    estimator :math:`q(\theta | x)` over pairs :math:`(\theta^*, x^*) \sim p(\theta, x)`.

    Equivalent to :func:`expected_coverage_mc` but the proportions :math:`r^*` are
    approximated by numerical integration over the domain, which is useful when the
    posterior estimator can be evaluated but not be sampled from.

    Arguments:
        log_p: A log-posterior estimator :math:`\log q(\theta | x)`.
        pairs: An iterable of pairs :math:`(\theta^*, x^*) \sim p(\theta, x)`.
        domain: A pair of lower and upper domain bounds for :math:`\theta`.
        device: The device on which the computations are performed.
        kwargs: Keyword arguments passed to :func:`lampe.utils.gridapply`.

    Returns:
        A vector of increasing credible levels and their respective expected coverages.

    Example:
        >>> domain = (torch.zeros(3), torch.ones(3))
        >>> prior = lampe.distributions.BoxUniform(*domain)
        >>> ratio = lampe.inference.NRE(3, 4)
        >>> log_p = lambda theta, x: ratio(theta, x) + prior.log_prob(theta)
        >>> testset = lampe.data.H5Dataset('test.h5')
        >>> levels, coverages = expected_coverage_ni(log_p, testset, domain)
    """

    if device is not None:
        domain = tuple(bound.to(device) for bound in domain)

    ranks = []

    with torch.no_grad():
        for theta, x in tqdm(pairs, unit='pair'):
            if device is not None:
                theta, x = theta.to(device), x.to(device)

            _, log_ps = gridapply(lambda theta: log_p(theta, x), domain, **kwargs)
            mask = log_p(theta, x) < log_ps
            rank = log_ps[mask].logsumexp(dim=0) - log_ps.flatten().logsumexp(dim=0)

            ranks.append(rank.exp())

    ranks = torch.stack(ranks).cpu()
    ranks = torch.cat((ranks, ranks.new_tensor((0.0, 1.0))))

    return (
        torch.sort(ranks).values,
        torch.linspace(0, 1, len(ranks)),
    )
