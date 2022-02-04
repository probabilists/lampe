r"""Training helpers"""

import numpy as np
import torch
import torch.nn as nn

from numpy import ndarray as Array
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm
from typing import *


def collect(
    pipe: Callable,  # embedding, estimator, criterion, ...
    loader: Iterable,
    optimizer: Optimizer = None,
    grad_clip: float = None,
) -> Array:
    r"""Sends loader's data through a pipe and collects the results.
    Optionally performs gradient descent steps."""

    results = []

    for data in loader:
        result = pipe(*data) if type(data) is tuple else pipe(data)
        results.append(result.tolist())

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

    return np.array(results)


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
                loss=np.nanmean(losses),
                lr=max(optimizer.lrs()),
            )

            yield epoch


def lrs(self) -> Iterator[float]:
    yield from (group['lr'] for group in self.param_groups)

setattr(Optimizer, 'lrs', lrs)


def parameters(self) -> Iterator[Tensor]:
    yield from (p for group in self.param_groups for p in group['params'])

setattr(Optimizer, 'parameters', parameters)
