r"""Monte Carlo Markov chain (MCMC) components."""

__all__ = [
    'MetropolisHastings',
]

import torch

from itertools import islice
from torch import Tensor
from torch.distributions import Distribution
from typing import *
from zuko.distributions import DiagNormal


class MetropolisHastings(object):
    r"""Creates a batched Metropolis-Hastings sampler.

    Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) sampling algorithm used to
    sample from intractable distributions :math:`p(x)` whose density is proportional to
    a tractable function :math:`f(x)`, with :math:`x \in \mathcal{X}`. The algorithm
    consists in repeating the following routine for :math:`t = 1` to :math:`T`, where
    :math:`x_0` is the initial sample and :math:`q(x' | x)` is a pre-defined transition
    distribution.

    .. math::
        1. ~ & x' \sim q(x' | x_{t-1}) \\
        2. ~ & \alpha \gets \frac{f(x')}{f(x_{t-1})}
            \frac{q(x_{t-1} | x')}{q(x' | x_{t-1})} \\
        3. ~ & u \sim \mathcal{U}(0, 1) \\
        4. ~ & x_t \gets \begin{cases}
            x' & \text{if } u \leq \alpha \\
            x_{t-1} & \text{otherwise}
        \end{cases}

    Asymptotically, i.e. when :math:`T \to \infty`, the distribution of samples
    :math:`x_t` is guaranteed to converge towards :math:`p(x)`. In this implementation,
    a Gaussian transition :math:`q(x' | x) = \mathcal{N}(x'; x, \Sigma)` is used, which
    can be modified by sub-classing :class:`MetropolisHastings`.

    Wikipedia:
        https://wikipedia.org/wiki/Metropolis-Hastings_algorithm

    Arguments:
        x_0: A batch of initial points :math:`x_0`, with shape :math:`(*, L)`.
        f: A function :math:`f(x)` proportional to a density function :math:`p(x)`.
        log_f: The logarithm :math:`\log f(x)` of a function proportional
            to :math:`p(x)`.
        sigma: The standard deviation of the Gaussian transition.
            Either a scalar or a vector.

    Example:
        >>> x_0 = torch.randn(128, 7)
        >>> log_f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> sampler = MetropolisHastings(x_0, log_f=log_f, sigma=0.5)
        >>> samples = [x for x in sampler(256, burn=128, step=4)]
        >>> samples = torch.stack(samples)
        >>> samples.shape
        torch.Size([32, 128, 7])
    """

    def __init__(
        self,
        x_0: Tensor,
        f: Callable[[Tensor], Tensor] = None,
        log_f: Callable[[Tensor], Tensor] = None,
        sigma: Union[float, Tensor] = 1.0,
    ):
        super().__init__()

        self.x_0 = x_0

        assert f is not None or log_f is not None, "Either 'f' or 'log_f' has to be provided."

        if f is None:
            self.log_f = log_f
        else:
            self.log_f = lambda x: f(x).log()

        self.sigma = sigma

    def q(self, x: Tensor) -> Distribution:
        return DiagNormal(x, torch.ones_like(x) * self.sigma)

    @property
    def symmetric(self) -> bool:
        return True

    def __iter__(self) -> Iterator[Tensor]:
        x = self.x_0

        # log f(x)
        log_f_x = self.log_f(x)

        while True:
            # y ~ q(y | x)
            y = self.q(x).sample()

            # log f(y)
            log_f_y = self.log_f(y)

            #     f(y)   q(x | y)
            # a = ---- * --------
            #     f(x)   q(y | x)
            log_a = log_f_y - log_f_x

            if not self.symmetric:
                log_a = log_a + self.q(y).log_prob(x) - self.q(x).log_prob(y)

            a = log_a.exp()

            # u in [0; 1]
            u = torch.rand(a.shape).to(a)

            # if u < a, x <- y
            # else x <- x
            mask = u < a

            x = torch.where(mask.unsqueeze(-1), y, x)
            log_f_x = torch.where(mask, log_f_y, log_f_x)

            yield x

    def __call__(self, stop: int, burn: int = 0, step: int = 1) -> Iterable[Tensor]:
        return islice(self, burn, stop, step)
