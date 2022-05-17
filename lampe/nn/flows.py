r"""Flows and parametric distributions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, Size
from typing import *

from . import MLP, MaskedMLP
from ..distributions import *
from ..utils import broadcast


__all__ = [
    'DistributionModule', 'TransformModule', 'FlowModule',
    'MaskedAutoregressiveTransform', 'MAF',
]


class DistributionModule(nn.Module):
    r"""Abstract distribution module."""

    def forward(y: Tensor = None) -> Distribution:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A distribution :math:`p(x | y)`.
        """

        raise NotImplementedError()


class TransformModule(nn.Module):
    r"""Abstract transform module."""

    def forward(y: Tensor = None) -> Transform:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A transform :math:`y = f(x | y)`.
        """

        raise NotImplementedError()


class FlowModule(DistributionModule):
    r"""Creates a normalizing flow module.

    Arguments:
        transforms: A list of transforms.
        base: A distribution.
    """

    def __init__(
        self,
        transforms: List[TransformModule],
        base: DistributionModule,
    ):
        super().__init__()

        self.transforms = nn.ModuleList(transforms)
        self.base = base

    def forward(self, y: Tensor = None) -> NormalizingFlow:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A normalizing flow :math:`p(x | y)`.
        """
        return NormalizingFlow(
            [t(y) for t in self.transforms],
            self.base(y) if y is None else self.base(y).expand(y.shape[:-1]),
        )


class Buffer(nn.Module):
    r"""Creates a buffer module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *args,
    ):
        super().__init__()

        self.meta = meta

        for i, arg in enumerate(args):
            self.register_buffer(f'_{i}', arg)

    def __repr__(self) -> str:
        return repr(self.forward())

    def forward(self, y: Tensor = None) -> Any:
        return self.meta(*self._buffers.values())


class Parametrization(nn.Module):
    r"""Creates a parametrization module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *shapes: Size,
        context: int = 0,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta
        self.shapes = list(map(Size, shapes))
        self.sizes = [s.numel() for s in self.shapes]

        if context > 0:
            self.params = build(context, sum(self.sizes), **kwargs)
        else:
            self.params = nn.ParameterList(list(
                map(nn.Parameter, map(torch.randn, self.shapes))
            ))

    def extra_repr(self) -> str:
        base = self.meta(*map(torch.randn, self.shapes))
        return f'(base): {base}'

    def forward(self, y: Tensor = None) -> Any:
        if isinstance(self.params, nn.ParameterList):
            args = self.params
        else:
            args = self.params(y).split(self.sizes, dim=-1)
            args = [a.reshape(a.shape[:-1] + s) for a, s in zip(args, self.shapes)]

        return self.meta(*args)


class MaskedAutoregressiveTransform(TransformModule):
    r"""Creates a masked autoregressive transform.

    Arguments:
        features: The number of features.
        context: The number of context features.
        passes: The number of passes for the inverse transformation.
        permute: Whether to randomly permute the features or not.
        base: The type of base transform (:py:`'affine'` or :py:`'spline'`).
        kwargs: Keyword arguments passed to :class:`lampe.nn.MaskedMLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = 2,  # coupling
        permute: bool = True,
        base: str = 'affine',
        **kwargs,
    ):
        super().__init__()

        if base == 'spline':
            bins = kwargs.pop('bins', 8)

            self.base = MonotonicRationalQuadraticSplineTransform
            self.shapes = [(bins,), (bins,), (bins - 1,)]
        else:  # if base == 'affine'
            self.base = MonotonicAffineTransform
            self.shapes = [(), ()]

        self.shapes = list(map(Size, self.shapes))
        self.sizes = [s.numel() for s in self.shapes]

        self.passes = passes if passes > 0 else features
        self.order = (torch.randperm if permute else torch.arange)(features) % self.passes

        in_order = torch.cat((self.order, torch.full((context,), -1)))
        out_order = self.order.tile(sum(self.sizes))

        self.params = MaskedMLP(in_order[..., None] < out_order, **kwargs)

    def extra_repr(self) -> str:
        base = self.base(*map(torch.randn, self.shapes))

        return '\n'.join([
            f'(base): {base}',
            f'(order): {self.order.tolist()}',
        ])

    def forward(self, y: Tensor = None) -> AutoregressiveTransform:
        def meta(x: Tensor) -> Transform:
            if y is not None:
                x = torch.cat(broadcast(x, y, ignore=1), dim=-1)

            params = self.params(x)
            params = params.reshape(*params.shape[:-1], sum(self.sizes), -1)
            params = params.transpose(-1, -2).contiguous()

            args = params.split(self.sizes, dim=-1)
            args = [a.reshape(a.shape[:-1] + s) for a, s in zip(args, self.shapes)]

            return self.base(*args)

        return AutoregressiveTransform(meta, self.passes)


class MAF(FlowModule):
    r"""Creates a masked autoregressive flow (MAF).

    References:
        Masked Autoregressive Flow for Density Estimation
        (Papamakarios et al., 2017)
        https://arxiv.org/abs/1705.07057

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transforms.
        linear: Whether to insert intermediate linear transforms or not.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        linear: bool = False,
        **kwargs,
    ):
        series = []

        for _ in range(transforms):
            if linear:
                series.extend([
                    Parametrization(MonotonicAffineTransform, (features,), (features,)),
                    Parametrization(
                        LULinearTransform,
                        (features * (features - 1) // 2,),
                        (features * (features - 1) // 2,),
                    ),
                ])

            series.append(MaskedAutoregressiveTransform(features, context, **kwargs))

        base = Buffer(DiagNormal, torch.zeros(features), torch.ones(features))

        super().__init__(series, base)
