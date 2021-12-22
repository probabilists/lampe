r"""Flows and parametric distributions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as D
import pyro.distributions.transforms as T

from functools import cached_property
from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule

from torch import Tensor
from typing import Callable, Union


class Modulify(nn.Module):
    r"""Module wrapper for non-module classes, like distributions or transforms

    Example:
        >>> dist = Modulify(D.Normal, loc=torch.tensor(0.), scale=torch.tensor(1.))
        >>> dist.obj.sample().device
        device(type='cpu')
        >>> dist.to('cuda')
        >>> dist.obj.sample().device
        device(type='cuda')
    """

    def __init__(
        self,
        constructor: Callable,
        *args,
        **kwargs,
    ):
        super().__init__()

        self._constructor = constructor
        self._args = args
        self._kwargs = kwargs

        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                self.register_buffer(f'__{i}', arg)

        for k, v in kwargs.items():
            if torch.is_tensor(v):
                self.register_buffer(f'__{k}', v)

    @cached_property
    def obj(self):
        args = [
            self.get_buffer(f'__{i}') if torch.is_tensor(arg) else arg
            for i, arg in enumerate(self._args)
        ]

        kwargs = {
            k: self.get_buffer(f'__{k}') if torch.is_tensor(v) else v
            for k, v in self._kwargs.items()
        }

        return self._constructor(*args, **kwargs)

    def _apply(self, *args, **kwargs):  # -> self
        if 'obj' in self.__dict__:
            del self.__dict__['obj']

        return super()._apply(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.obj(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        return repr(self.obj)


class ConstantConditionalTransformModule(ConditionalTransformModule):
    r"""Transform that is constant with respect to context

    References:
        https://docs.pyro.ai/en/stable/_modules/pyro/distributions/conditional.html
    """

    def __init__(self, transform: Union[TransformModule, Modulify]):
        super().__init__()

        self.transform = transform

    def condition(self, context: Tensor = None) -> TransformModule:
        if isinstance(self.transform, Modulify):
            return self.transform.obj
        return self.transform

    def clear_cache(self) -> None:
        self.condition().clear_cache()


class DistributionModule(Modulify):
    def log_prob(self, *args, **kwargs):
        return self.obj.log_prob(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.obj.sample(*args, **kwargs)


class NormalizingFlow(nn.Module):
    r"""Normalizing Flow

    (x, y) -> log p(x | y)

    Args:
        base: The base distribution.
        transforms: A list of (learnable) conditional transforms.
    """

    def __init__(
        self,
        base: DistributionModule,
        transforms: list[Union[TransformModule, Modulify]],
    ):
        super().__init__()

        self.base = base
        self.transforms = nn.ModuleList([
            t
            if isinstance(t, ConditionalTransformModule)
            else ConstantConditionalTransformModule(t)
            for t in transforms
        ])

    def condition(self, context: Tensor) -> D.TransformedDistribution:
        return D.TransformedDistribution(
            self.base.obj,
            [t.condition(context) for t in self.transforms],
        )


class MAF(NormalizingFlow):
    r"""Masked Autoregressive Flow (MAF)

    (x, y) -> log p(x | y)

    Args:
        input_size: The input size.
        context_size: The context size.
        num_transforms: The number of transforms.
        moments: The input moments (mu, sigma) for standardization.

        **kwargs are passed to `T.conditional_affine_autoregressive`.

    References:
        [1] Masked Autoregressive Flow for Density Estimation
        (Papamakarios et al., 2017)
        https://arxiv.org/abs/1705.07057
    """

    def __init__(
        self,
        input_size: int,
        context_size: int,
        num_transforms: int = 3,
        moments: tuple[Tensor, Tensor] = None,
        **kwargs,
    ):
        kwargs.setdefault('hidden_dims', [64, 64])

        base = DistributionModule(
            D.Normal,
            loc=torch.zeros(input_size),
            scale=torch.ones(input_size),
        )

        transforms = []

        if moments is not None:
            mu, sigma = moments
            transforms.append(
                Modulify(T.AffineTransform, loc=-mu / sigma, scale=1 / sigma)
            )

        for i in range(num_transforms):
            transforms.extend([
                T.conditional_affine_autoregressive(input_size, context_size, **kwargs),
                Modulify(T.Permute, torch.randperm(input_size)),
            ])

        super().__init__(base, transforms)
