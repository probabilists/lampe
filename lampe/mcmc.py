r"""Markov chain Monte Carlo (MCMC) samplers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
)

from torch import Tensor
from typing import Union


class MCMC:
    r"""Abstract Markov chain Monte Carlo (MCMC) algorithm"""

    def reference(self) -> Tensor:
        r""" x_0 """

        raise NotImplementedError()

    def f(self, x: Tensor) -> Tensor:
        r""" f(x) """

        return self.log_f(x).exp()

    def log_f(self, x: Tensor) -> Tensor:
        r""" log f(x) """

        return NotImplementedError()

    def __iter__(self):  # -> Tensor
        r""" x_i ~ p(x) ∝ f(x) """

        raise NotImplementedError()

    def __call__(
        self,
        steps: int,
        burn: int = 0,
        skip: int = 1,
        groupby: int = 1,
    ):  # -> Tensor
        r""" (x_0, x_1, ..., x_n) ~ p(x) """

        seq = islice(self, burn, steps, skip)

        if groupby > 1:
            buff = []

            for x in seq:
                buff.append(x)

                if len(buff) == groupby:
                    yield torch.cat(buff)
                    buff = []

            if buff:
                yield torch.cat(buff)
        else:
            yield from seq

    @torch.no_grad()
    def grid(
        self,
        bins: Union[int, list[int], Tensor],
        low: Tensor,
        high: Tensor,
        density: bool = False,
    ) -> Tensor:
        r"""Evaluate f(x) for all x in grid"""

        x = self.reference()

        # Shape
        B, D = x.shape

        if type(bins) is int:
            bins = [bins] * D

        # Create grid
        volume = 1.  # volume of one cell
        domains = []

        for l, h, b in zip(low, high, bins):
            step = (h - l) / b
            volume = volume * step

            dom = torch.linspace(l, h - step, b).to(step) + step / 2.
            domains.append(dom)

        grid = torch.stack(torch.meshgrid(*domains), dim=-1)
        grid = grid.view(-1, D).to(x)

        # Evaluate f(x) on grid
        f = []

        for x in grid.split(B):
            b = len(x)
            if b < B:
                x = F.pad(x, (0, 0, 0, B - b))

            f.append(self.f(x)[:b])

        f = torch.cat(f).view(bins)

        if density:
            f = f * volume

        return f


class MetropolisHastings(MCMC):
    r"""Abstract Metropolis-Hastings algorithm

    Wikipedia:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    """

    def __init__(self, sigma: Tensor = 1.):
        super().__init__()

        self.sigma = sigma
        self.q_symmetric = True

    def q(self, x: Tensor) -> Distribution:
        r"""Gaussian transition centered around x"""

        return Independent(Normal(x, torch.ones_like(x) * self.sigma), 1)

    @torch.no_grad()
    def __iter__(self):  # -> Tensor
        r""" x_i ~ p(x) ∝ f(x) """

        x = self.reference()

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

            if not self.q_symmetric:
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


class HamiltonianMonteCarlo(MCMC):
    r"""Abstract Hamiltonian Monte Carlo algorithm

    Wikipedia:
        https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo
    """
    pass  # TODO


class PESampler(MetropolisHastings):
    r"""Posterior Estimator (PE) sampler"""

    def __init__(
        self,
        estimator: nn.Module,
        prior: Distribution,
        x: Tensor,
        batch_size: int = 2 ** 10,  # 1024
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.prior = prior

        self.batch_shape = (batch_size,)
        self.x = x.expand(self.batch_shape + x.shape)

    def reference(self) -> Tensor:
        r""" theta_0 """

        return self.prior.sample(self.batch_shape)

    def log_f(self, theta: Tensor) -> Tensor:
        r""" log p(theta | x) """

        return self.estimator(theta, self.x)


class LRESampler(PESampler):
    r"""Likelihood-to-evidence Ratio Estimator (LRE) sampler"""

    def log_f(self, theta: Tensor) -> Tensor:
        r""" log r(theta, x) + log p(theta) """

        return self.estimator(theta, self.x) + self.prior.log_prob(theta)
