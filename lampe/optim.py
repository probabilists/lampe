r"""Optimizers and schedulers"""

import torch
import torch.nn as nn

from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from torch import Tensor
from typing import Union


def step(self, *args, **kwargs):
    return self._step()

setattr(Scheduler, '_step', Scheduler.step)
setattr(Scheduler, 'step', step)


def lrs(self) -> list[float]:
    return [group['lr'] for group in self.optimizer.param_groups]

setattr(Scheduler, 'lrs', property(lrs))


class ReduceLROnPlateau(Scheduler):
    r"""Reduce learning rate when a metric has stopped improving"""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.5,  # <= 1
        patience: int = 7,
        cooldown: int = 1,
        threshold: float = 1e-2,
        mode: str = 'minimize',  # 'maximize'
        min_lr: Union[float, list[float]] = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.gamma = gamma
        self.patience = patience
        self.cooldown = cooldown
        self.threshold = threshold
        self.mode = mode

        if type(min_lr) is float:
            min_lr = [min_lr] * len(optimizer.param_groups)
        self.min_lrs = min_lr

        self.best = self.worst # best metric so far
        self.last_best = last_epoch
        self.last_reduce = last_epoch

        super().__init__(optimizer, last_epoch, verbose)

    @property
    def worst(self):
        return float('-inf') if self.mode == 'maximize' else float('inf')

    def step(self, metric: float = None):
        self._current = self.worst if metric is None else metric
        return super().step()

    def get_lr(self):
        if self.mode == 'maximize':
            accept = self._current >= self.best * (1 + self.threshold)
        else:  # mode == 'minimize'
            accept = self._current <= self.best * (1 - self.threshold)

        if accept:
            self.best = self._current
            self.last_best = self.last_epoch

            return self.lrs

        if self.last_epoch - max(self.last_best, self.last_reduce + self.cooldown) <= self.patience:
            return self.lrs

        self.last_reduce = self.last_epoch

        return [
            max(lr * self.gamma, min_lr)
            for lr, min_lr in zip(self.lrs, self.min_lrs)
        ]
