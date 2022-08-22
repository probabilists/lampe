r"""General purpose helpers."""

import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from torch import Tensor
from torch.optim import Optimizer
from typing import *


@torch.no_grad()
def bisection(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 16,
) -> Tensor:
    r"""Applies the bisection method to find a root :math:`x` of a function
    :math:`f(x)` between the bounds :math:`a` an :math:`b`.

    Wikipedia:
        https://wikipedia.org/wiki/Bisection_method

    Arguments:
        f: A univariate function :math:`f(x)`.
        a: The bound :math:`a` such that :math:`f(a) \leq 0`.
        b: The bound :math:`b` such that :math:`0 \leq f(b)`.
        n: The number of iterations.

    Example:
        >>> f = torch.cos
        >>> a = torch.tensor(2.0)
        >>> b = torch.tensor(1.0)
        >>> bisection(f, a, b, n=16)
        tensor(1.5708)
    """

    for _ in range(n):
        c = (a + b) / 2

        mask = f(c) < 0

        a = torch.where(mask, c, a)
        b = torch.where(mask, b, c)

    return (a + b) / 2


def broadcast(*tensors: Tensor, ignore: Union[int, List[int]] = 0) -> Tuple[Tensor, ...]:
    r"""Broadcasts tensors together.

    The term broadcasting describes how PyTorch treats tensors with different shapes
    during arithmetic operations. In short, if possible, dimensions that have
    different sizes are expanded (without making copies) to be compatible.

    Arguments:
        ignore: The number(s) of dimensions not to broadcast.

    Example:
        >>> x = torch.rand(3, 1, 2)
        >>> y = torch.rand(4, 5)
        >>> x, y = broadcast(x, y, ignore=1)
        >>> x.shape
        torch.Size([3, 4, 2])
        >>> y.shape
        torch.Size([3, 4, 5])
    """

    if type(ignore) is int:
        ignore = [ignore] * len(tensors)

    dims = [t.dim() - i for t, i in zip(tensors, ignore)]
    common = torch.broadcast_shapes(*(t.shape[:i] for t, i in zip(tensors, dims)))

    return tuple(
        torch.broadcast_to(t, common + t.shape[i:])
        for t, i in zip(tensors, dims)
    )


def deepto(obj: Any, *args, **kwargs) -> Any:
    r"""Moves and/or casts all tensors referenced in an object, recursively.

    .. warning::
        Unless a tensor already has the correct type and device, the referenced
        tensors are replaced by a copy with the desired properties, which will
        break cross-references. Proceed with caution.

    Arguments:
        obj: An object.
        args: Positional arguments passed to :func:`torch.Tensor.to`.
        kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

    Example:
        >>> tensors = [torch.arange(i) for i in range(1, 4)]
        >>> deepto(tensors, dtype=torch.float)
        [tensor([0.]), tensor([0., 1.]), tensor([0., 1., 2.])]
    """

    if torch.is_tensor(obj):
        obj = obj.to(*args, **kwargs)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = deepto(value, *args, **kwargs)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = deepto(value, *args, **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(deepto(value, *args, **kwargs) for value in obj)
    elif hasattr(obj, '__dict__'):
        deepto(obj.__dict__, *args, **kwargs)

    return obj


class _AttachLimits(torch.autograd.Function):
    r"""Attaches the limits of integration to the computational graph."""

    @staticmethod
    def forward(
        ctx,
        f: Callable[[Tensor], Tensor],
        a: Tensor,
        b: Tensor,
        area: Tensor,
    ) -> Tensor:
        ctx.f = f
        ctx.save_for_backward(a, b)

        return area

    @staticmethod
    def backward(ctx, grad_area: Tensor) -> Tuple[Tensor, ...]:
        f = ctx.f
        a, b = ctx.saved_tensors

        if ctx.needs_input_grad[1]:
            grad_a = -f(a) * grad_area
        else:
            grad_a = None

        if ctx.needs_input_grad[2]:
            grad_b = f(b) * grad_area
        else:
            grad_b = None

        return None, grad_a, grad_b, grad_area


def gauss_legendre(
    f: Callable[[Tensor], Tensor],
    a: Tensor,
    b: Tensor,
    n: int = 3,
) -> Tensor:
    r"""Estimates the definite integral of :math:`f` from :math:`a` to :math:`b`
    using a :math:`n`-point Gauss-Legendre quadrature.

    .. math:: \int_a^b f(x) ~ dx \approx (b - a) \sum_{i = 1}^n w_i f(x_i)

    Wikipedia:
        https://wikipedia.org/wiki/Gauss-Legendre_quadrature

    Arguments:
        f: A univariate function :math:`f(x)`.
        a: The lower limit :math:`a`.
        b: The upper limit :math:`b`.
        n: The number of points :math:`n` at which the function is evaluated.

    Example:
        >>> f = lambda x: torch.exp(-x**2)
        >>> a, b = torch.tensor([-0.69, 4.2])
        >>> gauss_legendre(f, a, b, n=16)
        tensor(1.4807)
    """

    nodes, weights = leggauss(n, dtype=a.dtype, device=a.device)
    nodes = torch.lerp(
        a[..., None].detach(),
        b[..., None].detach(),
        nodes,
    ).movedim(-1, 0)

    area = (b - a).detach() * torch.tensordot(weights, f(nodes), dims=1)

    return _AttachLimits.apply(f, a, b, area)


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
    bins: Union[int, List[int]],
    bounds: Tuple[Tensor, Tensor],
    batch_size: int = 2**12,  # 4096
) -> Tuple[Tensor, Tensor]:
    r"""Evaluates a function :math:`f(x)` over a multi-dimensional domain split
    into grid cells. Instead of evaluating the function cell by cell, batches are
    given to the function.

    Arguments:
        f: A function :math:`f(x)`.
        bins: The number(s) of bins per dimension.
        bounds: A tuple of lower and upper domain bounds.
        batch_size: The size of the batches given to the function.

    Returns:
        The domain grid and the corresponding values.

    Example:
        >>> f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> lower, upper = torch.zeros(3), torch.ones(3)
        >>> x, y = gridapply(f, bins=10, bounds=(lower, upper))
        >>> x.shape
        torch.Size([10, 10, 10, 3])
        >>> y.shape
        torch.Size([10, 10, 10])
    """

    lower, upper = bounds

    # Shape
    dims = len(lower)

    if type(bins) is int:
        bins = [bins] * dims

    # Create grid
    domains = []

    for l, u, b in zip(lower, upper, bins):
        step = (u - l) / b
        dom = torch.linspace(l, u - step, b).to(step) + step / 2
        domains.append(dom)

    grid = torch.cartesian_prod(*domains)

    # Evaluate f(x) on grid
    y = [f(x) for x in grid.split(batch_size)]
    y = torch.cat(y)

    return grid.reshape(*bins, -1), y.reshape(*bins, *y.shape[1:])


@lru_cache(maxsize=None)
def leggauss(n: int, **kwargs) -> Tuple[Tensor, Tensor]:
    r"""Returns the nodes and weights for a :math:`n`-point Gauss-Legendre
    quadrature over the interval :math:`[0, 1]`.

    See :func:`numpy.polynomial.legendre.leggauss`.

    Arguments:
        n: The number of points :math:`n`.

    Example:
        >>> nodes, weights = leggauss(3)
        >>> nodes
        tensor([0.1127, 0.5000, 0.8873])
        >>> weights
        tensor([0.2778, 0.4444, 0.2778])
    """

    nodes, weights = np.polynomial.legendre.leggauss(n)

    nodes = (nodes + 1) / 2
    weights = weights / 2

    kwargs.setdefault('dtype', torch.get_default_dtype())

    return (
        torch.as_tensor(nodes, **kwargs),
        torch.as_tensor(weights, **kwargs),
    )
