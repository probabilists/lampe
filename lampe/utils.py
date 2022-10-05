r"""General purpose helpers."""

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.optim import Optimizer
from typing import *


class GDStep(object):
    r"""Creates a callable that performs gradient descent (GD) optimization steps
    for parameters :math:`\phi` with respect to differentiable loss values.

    The callable takes a scalar loss :math:`l` as input, performs a step

    .. math:: \phi \gets \text{GD}(\phi, \nabla_{\!\phi} \, l)

    and returns the loss, detached from the computational graph. To prevent invalid
    parameters, steps are skipped if not-a-number (NaN) or infinite values are found
    in the gradient. This feature requires CPU-GPU synchronization, which could be a
    bottleneck for some applications.

    Arguments:
        optimizer: An optimizer instance (e.g. :class:`torch.optim.SGD`).
        clip: The norm at which the gradients are clipped. If :py:`None`,
            gradients are not clipped.
    """

    def __init__(self, optimizer: Optimizer, clip: float = None):

        self.optimizer = optimizer
        self.parameters = [
            p
            for group in optimizer.param_groups
            for p in group['params']
        ]
        self.clip = clip

    def __call__(self, loss: Tensor) -> Tensor:
        if loss.isfinite().all():
            self.optimizer.zero_grad()
            loss.backward()

            if self.clip is None:
                self.optimizer.step()
            else:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if norm.isfinite():
                    self.optimizer.step()

        return loss.detach()


def gridapply(
    f: Callable[[Tensor], Tensor],
    domain: Tuple[Tensor, Tensor],
    bins: Union[int, List[int]] = 128,
    batch_size: int = 4096,
) -> Tuple[Tensor, Tensor]:
    r"""Evaluates a function :math:`f(x)` over a multi-dimensional domain split
    into grid cells. Instead of evaluating the function cell by cell, batches are
    given to the function.

    Arguments:
        f: A function :math:`f(x)`.
        domain: A pair of lower and upper domain bounds.
        bins: The number(s) of bins per dimension.
        batch_size: The size of the batches given to the function.

    Returns:
        The domain grid and the corresponding values.

    Example:
        >>> f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> lower, upper = torch.zeros(3), torch.ones(3)
        >>> x, y = gridapply(f, (lower, upper), bins=8)
        >>> x.shape
        torch.Size([8, 8, 8, 3])
        >>> y.shape
        torch.Size([8, 8, 8])
    """

    lower, upper = domain

    # Shape
    dims = len(lower)

    if type(bins) is int:
        bins = [bins] * dims

    # Create grid
    coordinates = []

    for l, u, b in zip(lower, upper, bins):
        step = (u - l) / b
        ticks = torch.linspace(l, u - step, b).to(step) + step / 2
        coordinates.append(ticks)

    grid = torch.cartesian_prod(*coordinates)

    # Evaluate f(x) on grid
    y = [f(x) for x in grid.split(batch_size)]
    y = torch.cat(y)

    return grid.reshape(*bins, -1), y.reshape(*bins, *y.shape[1:])
