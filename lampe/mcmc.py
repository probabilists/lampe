r"""Markov chain Monte Carlo (MCMC) samplers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from itertools import islice
from torch import Tensor
from typing import *

from .priors import Distribution, JointNormal


class MCMC(ABC):
    r"""Abstract Markov chain Monte Carlo (MCMC) algorithm"""

    def __init__(
        self,
        x_0: Tensor,  # x_0
        f: Callable = None,  # f(x)
        log_f: Callable = None,  # log f(x)
    ):
        super().__init__()

        self.x_0 = x_0

        assert f is not None or log_f is not None, \
            "either 'f' or 'log_f' must be provided"

        if f is None:
            self.f = lambda x: log_f(x).exp()
            self.log_f = log_f
        else:
            self.f = f
            self.log_f = lambda x: f(x).log()

    @abstractmethod
    def __iter__(self) -> Iterator[Tensor]:
        r""" x_i ~ p(x) ∝ f(x) """
        pass

    @torch.no_grad()
    def __call__(
        self,
        n: int,
        burn: int = 0,
        step: int = 1,
        groupby: int = 1,
    ) -> Iterator[Tensor]:
        r""" (x_1, ..., x_n) ~ p(x) """

        seq = islice(self, burn, burn + n * step, step)

        if groupby > 1:
            buff = []

            for x in seq:
                buff.append(x)

                if len(buff) == groupby:
                    yield torch.cat(buff)
                    buff.clear()

            if buff:
                yield torch.cat(buff)
        else:
            yield from seq

    @torch.no_grad()
    def grid(
        self,
        bins: Union[int, List[int]],
        bounds: Tuple[Tensor, Tensor],
    ) -> Tensor:
        r"""Evaluates f(x) for all x in grid"""

        x = self.x_0

        # Shape
        D = x.shape[-1]
        B = x.numel() // D

        if type(bins) is int:
            bins = [bins] * D

        # Create grid
        domains = []

        for l, u, b in zip(bounds[0], bounds[1], bins):
            step = (u - l) / b
            dom = torch.linspace(l, u - step, b).to(step) + step / 2.
            domains.append(dom)

        grid = torch.stack(torch.meshgrid(*domains, indexing='ij'), dim=-1)
        grid = grid.view(-1, D).to(x)

        # Evaluate f(x) on grid
        f = []

        for x in grid.split(B):
            b = len(x)

            if b < B:
                x = F.pad(x, (0, 0, 0, B - b))
                y = self.f(x)[:b]
            else:
                y = self.f(x)

            f.append(y)

        return torch.cat(f).view(bins)


class MetropolisHastings(MCMC):
    r"""Metropolis-Hastings algorithm

    Wikipedia:
        https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
    """

    def __init__(self, *args, sigma: Tensor = 1., **kwargs):
        super().__init__(*args, **kwargs)

        self.sigma = sigma

    def q(self, x: Tensor) -> Distribution:
        r"""Gaussian transition centered around x"""

        return JointNormal(x, torch.ones_like(x) * self.sigma)

    @property
    def symmetric(self) -> bool:
        r"""Whether q(x | y) is equal to q(y | x)"""

        return True

    def __iter__(self) -> Iterator[Tensor]:
        r""" x_i ~ p(x) ∝ f(x) """

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


class InferenceSampler(MetropolisHastings):
    r"""Inference MCMC sampler"""

    def __init__(
        self,
        x: Tensor,  # x
        prior: Distribution,  # p(theta)
        likelihood: Callable = None,  # log p(x | theta)
        posterior: Callable = None,  # log p(theta | x)
        ratio: Callable = None,  # log p(theta | x) - log p(theta)
        batch_size: int = 2**10,  # 1024
        **kwargs,
    ):
        theta_0 = prior.sample((batch_size,))
        x = x.expand((batch_size,) + x.shape)

        assert likelihood is not None or posterior is not None or ratio is not None, \
            "either 'likelihood', 'posterior' or 'ratio' must be provided"

        if likelihood is not None:
            log_f = lambda theta: likelihood(theta, x) + prior.log_prob(theta)
        elif posterior is not None:
            log_f = lambda theta: posterior(theta, x)
        elif ratio is not None:
            log_f = lambda theta: ratio(theta, x) + prior.log_prob(theta)

        super().__init__(
            x_0=theta_0,
            log_f=log_f,
            **kwargs,
        )
