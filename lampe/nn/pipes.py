r"""Pipelines"""

import torch
import torch.nn as nn

from torch import Tensor
from typing import *

from .masks import MaskSampler


class Pipe(nn.Module):
    r"""Abstract pipeline class"""

    def __init__(self, device: torch.device = None):
        super().__init__()

        self.register_buffer('dummy', torch.tensor(0., device=device))

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def move(self, *args) -> Tuple[Tensor]:
        return tuple(map(lambda x: x.to(self.device), args))


class NREPipe(Pipe):
    r"""NRE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.criterion = criterion
        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        theta_prime = torch.roll(theta, 1, dims=0)
        x = self.embedding(x)

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
        mask_smpl: MaskSampler,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.mask_smpl = mask_smpl
        self.criterion = criterion
        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        theta_prime = torch.roll(theta, 1, dims=0)
        x = self.embedding(x)
        mask = self.mask_smpl.sample(theta.shape[:-1])

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
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        x = self.embedding(x)
        log_prob = self.estimator(theta, x)

        return -log_prob.mean()


class AMNPEPipe(Pipe):
    r"""AMNPE training pipeline"""

    def __init__(
        self,
        estimator: nn.Module,
        mask_smpl: MaskSampler,
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.mask_smpl = mask_smpl
        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta, x = self.move(theta, x)

        x = self.embedding(x)
        mask = self.mask_smpl.sample(theta.shape[:-1])

        log_prob = self.estimator(theta, x, mask)

        return -log_prob.mean()
