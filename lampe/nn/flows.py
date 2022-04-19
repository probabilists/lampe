r"""Flows and parametric distributions.

.. admonition:: TODO

    * Finish documentation.
    * Drop :mod:`nflows`.
    * Find references.
"""

import nflows.distributions as D
import nflows.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows.flows import Flow
from torch import Tensor
from typing import *


class NormalizingFlow(Flow):
    r"""Creates a normalizing flow :math:`p_\phi(x | y)`.

    TODO

    Arguments:
        base: A base distribution.
        transforms: A list of parametric conditional transforms.
    """

    def __init__(self, base: D.Distribution, transforms: List[T.Transform]):
        super().__init__(
            T.CompositeTransform(transforms),
            base
        )

    def log_prob(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Returns the log-density :math:`\log p_\phi(x | y)`."""

        return super().log_prob(
            x.reshape(-1, x.shape[-1]),
            y.reshape(-1, y.shape[-1]),
        ).reshape(x.shape[:-1])

    @torch.no_grad()
    def sample(self, y: Tensor, shape: torch.Size = ()) -> Tensor:
        return self.rsample(y, shape)

    def rsample(self, y: Tensor, shape: torch.Size = ()) -> Tensor:
        r"""Samples from the conditional distribution :math:`p_\phi(x | y)`."""

        size = torch.Size(shape).numel()

        x = super()._sample(size, y.reshape(-1, y.shape[-1]))
        x = x.reshape(y.shape[:-1] + shape + x.shape[-1:])

        return x


class MAF(NormalizingFlow):
    r"""Creates a masked autoregressive flow (MAF).

    TODO

    References:
        Masked Autoregressive Flow for Density Estimation
        (Papamakarios et al., 2017)
        https://arxiv.org/abs/1705.07057

    Arguments:
        x_size: The input size.
        y_size: The context size.
        arch: The flow architecture.
        num_transforms: The number of transforms.
        moments: The input moments (mu, sigma) for standardization.
        kwargs: Keyword arguments passed to the transform.
    """

    def __init__(
        self,
        x_size: int,
        y_size: int,
        arch: str = 'affine',  # ['PRQ', 'UMNN']
        num_transforms: int = 5,
        lu_linear: bool = False,
        moments: Tuple[Tensor, Tensor] = None,
        **kwargs,
    ):
        kwargs.setdefault('hidden_features', 64)
        kwargs.setdefault('num_blocks', 2)
        kwargs.setdefault('use_residual_blocks', False)
        kwargs.setdefault('use_batch_norm', False)
        kwargs.setdefault('activation', F.elu)

        if arch == 'PRQ':
            kwargs['tails'] = 'linear'
            kwargs.setdefault('num_bins', 8)
            kwargs.setdefault('tail_bound', 3.)

            MAT = T.MaskedPiecewiseRationalQuadraticAutoregressiveTransform
        elif arch == 'UMNN':
            kwargs.setdefault('integrand_net_layers', [64, 64, 64])
            kwargs.setdefault('cond_size', 32)
            kwargs.setdefault('nb_steps', 32)

            MAT = T.MaskedUMNNAutoregressiveTransform
        else:  # arch == 'affine'
            MAT = T.MaskedAffineAutoregressiveTransform

        transforms = []

        if moments is not None:
            mu, sigma = moments
            transforms.append(T.PointwiseAffineTransform(-mu / sigma, 1 / sigma))

        for _ in range(num_transforms):
            transforms.extend([
                MAT(
                    features=x_size,
                    context_features=y_size,
                    **kwargs,
                ),
                T.RandomPermutation(features=x_size),
            ])

            if lu_linear:
                transforms.append(T.LULinear(features=x_size))

        base = D.StandardNormal((x_size,))

        super().__init__(base, transforms)
