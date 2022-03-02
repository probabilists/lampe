r"""Training helpers"""

import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm
from typing import *


def lrs(self) -> Iterator[float]:
    yield from (group['lr'] for group in self.param_groups)

setattr(Optimizer, 'lrs', lrs)


def parameters(self) -> Iterator[Tensor]:
    yield from (p for group in self.param_groups for p in group['params'])

setattr(Optimizer, 'parameters', parameters)


def collect(
    pipe: Callable,  # embedding, estimator, criterion, ...
    loader: Iterable,
    optimizer: Optimizer = None,
    grad_clip: float = None,
) -> Tensor:
    r"""Sends loader's data through a pipe and collects the results.
    Optionally performs gradient descent steps."""

    results = []

    for data in loader:
        result = pipe(*data) if type(data) is tuple else pipe(data)
        results.append(result.detach())

        if optimizer is None:
            continue

        loss = result.mean()
        if not loss.isfinite():
            continue

        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            norm = nn.utils.clip_grad_norm_(optimizer.parameters(), grad_clip)
            if not norm.isfinite():
                continue

        optimizer.step()

    return torch.stack(results)


def trainbar(
    epochs: int,
    pipe: Callable,
    loader: Iterable,
    optimizer: Optimizer,
    **kwargs,
) -> Iterator[int]:
    r"""Iterator over training epochs with a progress bar"""

    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:
            losses = collect(
                pipe,
                loader,
                optimizer,
                **kwargs,
            )

            tq.set_postfix(
                loss=torch.nanmean(losses).item(),
                lr=max(optimizer.lrs()),
            )

            yield epoch


class PlateauDetector(object):
    r"""Sequence abstraction to detect plateau"""

    def __init__(
        self,
        threshold: float = 1e-2,
        patience: int = 16,
        mode: str = 'min',  # 'max'
    ):
        self.threshold = threshold
        self.patience = patience
        self.mode = mode

        self.sequence = [float('+inf' if mode == 'min' else '-inf')]
        self.best_time = 0

    @property
    def time(self) -> int:
        return len(self.sequence) - 1

    @property
    def best(self) -> float:
        return self.sequence[self.best_time]

    def step(self, value: float) -> None:
        self.sequence.append(value)

        if self.mode == 'min':
            better = value < self.best * (1 - self.threshold)
        else:
            better = value > self.best * (1 + self.threshold)

        if better:
            self.best_time = self.time

    @property
    def plateau(self) -> bool:
        return self.time > self.best_time + self.patience
