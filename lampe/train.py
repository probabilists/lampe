r"""Training routines"""

import numpy as np
import torch
import torch.nn as nn

from numpy import ndarray as Array
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm
from typing import *


def train_epoch(
    pipe: Callable,  # embedding, estimator, criterion, ...
    loader: Iterable,
    optimizer: Optimizer,
    grad_clip: float = None,
) -> Array:
    r"""Performs a training epoch"""

    losses = []

    for inputs in loader:
        l = pipe(*inputs)

        losses.append(l.item())

        if not l.isfinite():
            continue

        optimizer.zero_grad()

        l.backward()

        if grad_clip is not None:
            norm = nn.utils.clip_grad_norm_(optimizer.parameters(), grad_clip)

            if not norm.isfinite():
                continue

        optimizer.step()

    return np.array(losses)


def train_loop(
    pipe: Callable,
    loader: Iterable,
    optimizer: Optimizer,
    epochs: Union[int, range],
    **kwargs,
) -> Iterator[Tuple[int, Array]]:
    r"""Loops over training epochs"""

    if type(epochs) is int:
        epochs = range(epochs)

    with tqdm(epochs, unit='epoch') as tq:
        for e in tq:
            losses = train_epoch(
                pipe,
                loader,
                optimizer,
                **kwargs,
            )

            tq.set_postfix(
                lr=max(optimizer.lrs()),
                loss=np.nanmean(losses),
            )

            yield e, losses


def lrs(self) -> Iterator[float]:
    yield from (group['lr'] for group in self.param_groups)

setattr(Optimizer, 'lrs', lrs)


def parameters(self) -> Iterator[Tensor]:
    yield from (p for group in self.param_groups for p in group['params'])

setattr(Optimizer, 'parameters', parameters)
