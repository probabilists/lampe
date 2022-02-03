r"""Pipelines"""

import torch
import torch.nn as nn

from torch import Tensor, BoolTensor
from torch.distributions import Distribution
from typing import *


class Pipe(nn.Module):
    r"""Abstract pipeline class"""

    def __init__(
        self,
        embedding: nn.Module = nn.Identity(),
        filtr: BoolTensor = None,
        device: torch.device = None,
    ):
        super().__init__()

        self.embedding = embedding

        if filtr is None:
            self.filtr = None
        else:
            self.register_buffer('filtr', filtr)

        self.register_buffer('dummy', torch.tensor(0., device=device))

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def move(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = theta.to(self.device), x.to(self.device)

        if self.filtr is not None:
            theta = theta[..., self.filtr]

        x = self.embedding(x)

        return theta, x


class NREPipe(Pipe):
    r"""NRE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.criterion = criterion

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        theta_prime = torch.roll(theta, 1, dims=0)

        ratio, ratio_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            torch.stack((x, x)),
        )

        l1 = self.criterion(ratio, torch.ones_like(ratio))
        l0 = self.criterion(ratio_prime, torch.zeros_like(ratio))

        return (l1 + l0) / 2


class AMNREPipe(Pipe):
    r"""AMNRE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.mask_dist = mask_dist
        self.criterion = criterion

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        theta_prime = torch.roll(theta, 1, dims=0)
        mask = self.mask_dist.sample(theta.shape[:-1])

        ratio, ratio_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            torch.stack((x, x)),
            mask,
        )

        l1 = self.criterion(ratio, torch.ones_like(ratio))
        l0 = self.criterion(ratio_prime, torch.zeros_like(ratio))

        return (l1 + l0) / 2


class NPEPipe(Pipe):
    r"""NPE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        log_prob = self.estimator(theta, x)

        return -log_prob.mean()


class AMNPEPipe(Pipe):
    r"""AMNPE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        mask = self.mask_dist.sample(theta.shape[:-1])

        log_prob = self.estimator(theta, x, mask)

        return -log_prob.mean()
