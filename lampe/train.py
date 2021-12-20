r"""Training routines"""

import torch
import torch.nn as nn

from itertools import islice
from time import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torch import Tensor
from typing import Iterable

from .nn import NLLLoss, BCEWithLogitsLoss
from .optim import Optimizer, Scheduler, ExponentialLR


class Trainer(object):
    r"""Trainer"""

    def __init__(
        self,
        pipe: nn.Module,  # embedding, model, criterion, etc.
        train_loader: Iterable,
        valid_loader: Iterable,
        optimizer: Optimizer,
        scheduler: Scheduler = None,
        clip: float = None,  # gradient norm clip threshold
        writer: SummaryWriter = None,
        graph: bool = False,
    ):
        super().__init__()

        self.pipe = pipe

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        if scheduler is None:
            scheduler = ExponentialLR(optimizer, 1)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip

        if writer is None:
            writer = SummaryWriter()
        self.writer = writer

        if graph:
            self.writer.add_graph(self.pipe, next(iter(self.train_loader)))

    @property
    def lr(self) -> float:
        return min(self.scheduler.lrs)

    @property
    def parameters(self) -> list[Tensor]:
        return [p for group in self.optimizer.param_groups for p in group['params']]

    @property
    def epoch(self) -> int:
        return self.scheduler.last_epoch

    def optimize(self) -> Tensor:
        r"""Optimization epoch"""

        self.pipe.train()

        losses = []

        for inputs in self.train_loader:
            l = self.pipe(*inputs)

            if not l.isfinite():
                continue

            losses.append(l.item())

            self.optimizer.zero_grad()

            l.backward()

            if self.clip is not None:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if not norm.isfinite():
                    continue

            self.optimizer.step()

        return torch.tensor(losses)

    @torch.no_grad()
    def validate(self) -> Tensor:
        r"""Validation epoch"""

        self.pipe.eval()

        losses = []

        for inputs in self.valid_loader:
            l = self.pipe(*inputs)

            if not l.isfinite():
                continue

            losses.append(l.item())

        return torch.tensor(losses)

    def __call__(self, epochs: int):
        r"""Training loop"""

        with tqdm(total=epochs, unit='epoch') as tq:
            tq.set_description('Epochs')

            for _ in range(epochs):
                self.writer.add_scalar('train/lr', self.lr, self.epoch)

                start = time()

                train_losses = self.optimize()
                valid_losses = self.validate()

                end = time()

                self.writer.add_scalar('train/time', end - start, self.epoch)
                self.writer.add_scalars('train/loss_mean', {
                    'train': train_losses.mean(),
                    'valid': valid_losses.mean(),
                }, self.epoch)
                self.writer.add_scalars('train/loss_median', {
                    'train': train_losses.median(),
                    'valid': valid_losses.median(),
                }, self.epoch)

                loss = valid_losses.mean().item()
                self.scheduler.step(metric=loss)

                tq.set_postfix(lr=self.lr, loss=loss)
                tq.update(1)

                yield self.epoch


class NREPipe(nn.Module):
    r"""NRE training pipeline"""

    def __init__(self, model: nn.Module, criterion: nn.Module = BCEWithLogitsLoss()):
        super().__init__()

        self.model = model
        self.criterion = criterion

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta_prime = torch.roll(theta, 1, 0)

        y = self.model.embedding(x)
        ratio, ratio_prime = self.model(
            torch.stack((theta, theta_prime)),
            torch.stack((y, y)),
        )

        return self.criterion(ratio, ratio_prime)


class NPEPipe(nn.Module):
    r"""NPE training pipeline"""

    def __init__(self, model: nn.Module, criterion: nn.Module = NLLLoss()):
        super().__init__()

        self.model = model
        self.criterion = criterion

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        y = self.model.embedding(x)
        log_prob = self.model(theta, y)

        return self.criterion(log_prob)
