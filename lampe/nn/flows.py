r"""Flows and parametric distributions."""

__all__ = [
    'DistributionModule',
    'TransformModule',
    'FlowModule',
    'MaskedAutoregressiveTransform',
    'MAF',
    'NSF',
    'NeuralAutoregressiveTransform',
    'UnconstrainedNeuralAutoregressiveTransform',
    'NAF',
]

import abc
import torch
import torch.nn as nn

from math import ceil
from torch import Tensor, Size
from typing import *

from . import MLP, MaskedMLP, MonotonicMLP
from ..distributions import *
from ..utils import broadcast


class DistributionModule(nn.Module, abc.ABC):
    r"""Abstract distribution module."""

    @abc.abstractmethod
    def forward(y: Tensor = None) -> Distribution:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A distribution :math:`p(x | y)`.
        """

        pass


class TransformModule(nn.Module, abc.ABC):
    r"""Abstract transform module."""

    @abc.abstractmethod
    def forward(y: Tensor = None) -> Transform:
        r"""
        Arguments:
            y: A context :math:`y`.

        Returns:
            A transform :math:`z = f(x | y)`.
        """

        pass


class FlowModule(DistributionModule):
    r"""Creates a normalizing flow module.

    Arguments:
        transforms: A list of transform modules.
        base: A distribution module.
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

        transforms = [t(y) for t in self.transforms]

        if y is None:
            base = self.base(y)
        else:
            base = self.base(y).expand(y.shape[:-1])

        return NormalizingFlow(transforms, base)


class Buffer(nn.Module):
    r"""Creates a buffer module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta

        for i, arg in enumerate(args):
            self.register_buffer(f'_{i}', arg)

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return repr(self.forward())

    def forward(self, y: Tensor = None) -> Any:
        return self.meta(*self._buffers.values(), **self.kwargs)


class Parameters(nn.ParameterList):
    r"""Holds tensors as parameters."""

    def __init__(self, parameters: Iterable[Tensor]):
        super().__init__(list(map(nn.Parameter, parameters)))

    def extra_repr(self) -> str:
        return '\n'.join(f'({i}): {p.shape}' for i, p in enumerate(self))


class Parametrization(nn.Module):
    r"""Creates a parametrization module."""

    def __init__(
        self,
        meta: Callable[..., Any],
        *params: Tensor,
        context: int = 0,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.meta = meta
        self.shapes = [p.shape for p in params]
        self.sizes = [s.numel() for s in self.shapes]

        if context > 0:
            self.params = build(context, sum(self.sizes), **kwargs)
        else:
            self.params = Parameters(params)

    def extra_repr(self) -> str:
        base = self.meta(*map(torch.randn, self.shapes))
        return f'(base): {base}'

    def forward(self, y: Tensor = None) -> Any:
        if isinstance(self.params, Parameters):
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
        passes: The number of passes for the inverse transformation. If :py:`None`,
            use the number of features instead.
        order: The feature ordering. If :py:`None`, use :py:`range(features)` instead.
        univariate: A univariate transformation constructor.
        shapes: The shapes of the univariate transformation parameters.
        kwargs: Keyword arguments passed to :class:`lampe.nn.MaskedMLP`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        passes: int = None,
        order: LongTensor = None,
        univariate: Callable[..., Transform] = MonotonicAffineTransform,
        shapes: List[Size] = [(), ()],
        **kwargs,
    ):
        super().__init__()

        self.univariate = univariate
        self.shapes = list(map(Size, shapes))
        self.sizes = [s.numel() for s in self.shapes]

        self.register_buffer('order', None)

        if passes is None:
            passes = features

        if order is None:
            order = torch.arange(features)

        self.passes = min(max(passes, 1), features)
        self.order = torch.div(order, ceil(features / self.passes), rounding_mode='floor')

        in_order = torch.cat((self.order, torch.full((context,), -1)))
        out_order = self.order.tile(sum(self.sizes))

        self.params = MaskedMLP(out_order[:, None] > in_order, **kwargs)

    def extra_repr(self) -> str:
        base = self.univariate(*map(torch.randn, self.shapes))

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

            return self.univariate(*args)

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
        randperm: Whether features are randomly permuted between transforms or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transforms.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        **kwargs,
    ):
        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            MaskedAutoregressiveTransform(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        base = Buffer(DiagNormal, torch.zeros(features), torch.ones(features))

        super().__init__(transforms, base)


class NSF(MAF):
    r"""Creates a neural spline flow (NSF).

    References:
        Neural Spline Flows
        (Durkan et al., 2019)
        https://arxiv.org/abs/1906.04032

    Arguments:
        features: The number of features.
        context: The number of context features.
        bins: The number of bins :math:`K`.
        kwargs: Keyword arguments passed to :class:`MAF`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        bins: int = 8,
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=MonotonicRQSTransform,
            shapes=[(bins,), (bins,), (bins - 1,)],
            **kwargs,
        )

        self.transforms.insert(0, Buffer(SoftclipTransform))
        self.transforms.append(Buffer(lambda: SoftclipTransform().inv))


class NeuralAutoregressiveTransform(MaskedAutoregressiveTransform):
    r"""Creates a neural autoregressive transform.

    The monotonic neural network is parametrized by its internal positive weights,
    which are independent of the features and context. To modulate its behavior, it
    receives as input a signal that is autoregressively dependent on the features
    and context.

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the monotonic network.
        network: Keyword arguments passed to :class:`lampe.nn.MonotonicMLP`.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 8,
        network: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=self.univariate,
            shapes=[(signal,)],
            **kwargs,
        )

        self.network = MonotonicMLP(1 + signal, 1, **network)

    def univariate(self, signal: Tensor) -> Transform:
        def f(x: Tensor) -> Tensor:
            return self.network(
                torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1)
            ).squeeze(dim=-1)

        return MonotonicTransform(f)


class UnconstrainedNeuralAutoregressiveTransform(MaskedAutoregressiveTransform):
    r"""Creates an unconstrained neural autoregressive transform.

    The integrand neural network is parametrized by its internal weights, which are
    independent of the features and context. To modulate its behavior, it receives as
    input a signal that is autoregressively dependent on the features and context. The
    integration constant has the same dependencies as the signal.

    Arguments:
        features: The number of features.
        context: The number of context features.
        signal: The number of signal features of the integrand network.
        network: Keyword arguments passed to :class:`lampe.nn.MLP`.
        kwargs: Keyword arguments passed to :class:`MaskedAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        signal: int = 8,
        network: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(
            features=features,
            context=context,
            univariate=self.univariate,
            shapes=[(signal,), ()],
            **kwargs,
        )

        network.setdefault('activation', 'ELU')

        self.integrand = MLP(1 + signal, 1, **network)
        self.integrand.add_module(
            str(len(self.integrand)),
            nn.Softplus(),
        )

    def univariate(self, signal: Tensor, constant: Tensor) -> Transform:
        def f(x: Tensor) -> Tensor:
            return self.integrand(
                torch.cat(broadcast(x[..., None], signal, ignore=1), dim=-1)
            ).squeeze(dim=-1)

        return UnconstrainedMonotonicTransform(f, constant)


class NAF(FlowModule):
    r"""Creates a neural autoregressive flow (NAF).

    References:
        Neural Autoregressive Flows
        (Huang et al., 2018)
        https://arxiv.org/abs/1804.00779

        Unconstrained Monotonic Neural Networks
        (Wehenkel et al., 2019)
        https://arxiv.org/abs/1908.05164

    Arguments:
        features: The number of features.
        context: The number of context features.
        transforms: The number of autoregressive transforms.
        randperm: Whether features are randomly permuted between transforms or not.
            If :py:`False`, features are in ascending (descending) order for even
            (odd) transforms.
        unconstrained: Whether to use unconstrained or regular monotonic networks.
        kwargs: Keyword arguments passed to :class:`NeuralAutoregressiveTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        transforms: int = 3,
        randperm: bool = False,
        unconstrained: bool = False,
        **kwargs,
    ):
        if unconstrained:
            build = UnconstrainedNeuralAutoregressiveTransform
        else:
            build = NeuralAutoregressiveTransform

        orders = [
            torch.arange(features),
            torch.flipud(torch.arange(features)),
        ]

        transforms = [
            build(
                features=features,
                context=context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                **kwargs,
            )
            for i in range(transforms)
        ]

        for i in reversed(range(len(transforms))):
            transforms.insert(i, Buffer(SoftclipTransform))

        base = Buffer(DiagNormal, torch.zeros(features), torch.ones(features))

        super().__init__(transforms, base)
