r"""Modules and layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from torch.distributions import Distribution
from typing import *

from .flows import MAF


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
}


class Broadcast(nn.Module):
    r"""Broadcast layer

    Args:
        keep: The number of dimensions to not broadcast
    """

    def __init__(self, keep: int = 0):
        super().__init__()

        self.keep = keep

    def split(self, shape: torch.Size) -> Tuple[torch.Size, torch.Size]:
        index = len(shape) - self.keep
        return shape[:index], shape[index:]

    def forward(self, *xs: Tensor) -> List[Tensor]:
        splits = [self.split(x.shape) for x in xs]

        before, after = zip(*splits)
        before = torch.broadcast_shapes(*before)

        return [
            torch.broadcast_to(x, before + a)
            for x, a in zip(xs, after)
        ]

    def extra_repr(self) -> str:
        return f'keep={self.keep}'


class Affine(nn.Module):
    r"""Element-wise affine layer

    Args:
        shift: The shift term
        scale: The scale factor
    """

    def __init__(self, shift: Tensor, scale: Tensor):
        super().__init__()

        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)

    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale + self.shift

    def extra_repr(self) -> str:
        return '\n'.join([
            f'(shift): {self.shift.cpu()}',
            f'(scale): {self.scale.cpu()}',
        ])


class BatchNorm0d(nn.BatchNorm1d):
    r"""Batch Normalization (BatchNorm) layer for scalars

    References:
        [1] Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        (Ioffe et al., 2015)
        https://arxiv.org/abs/1502.03167
    """

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape

        x = x.reshape(-1, shape[-1])
        x = super().forward(x)
        x = x.reshape(shape)

        return x


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        activation: The activation layer type.
        batchnorm: Whether to use batch normalization or not.
        dropout: The dropout rate.

        **kwargs are passed to `nn.Linear`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        activation: str = 'ReLU',
        batchnorm: bool = False,
        dropout: float = 0.,
        **kwargs,
    ):
        activation = ACTIVATIONS[activation]
        batchnorm = BatchNorm0d if batchnorm else lambda _: None
        dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers = []

        for before, after  in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                batchnorm(after),
                activation(),
                dropout,
            ])

        layers = layers[:-3]
        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


class ResBlock(MLP):
    r"""Residual Block (ResBlock)

    Args:
        features: The input, output and hidden features.
        block_layers: The number of block layers.

        **kwargs are passed to `MLP`.
    """

    def __init__(self, features: int, block_layers: int = 2, **kwargs):
        super().__init__(features, features, [features] * block_layers, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return input + super().forward(input)


class ResMLP(nn.Sequential):
    r"""Residual Multi-Layer Perceptron (ResMLP)

    Args:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.

        **kwargs are passed to `ResBlock`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        **kwargs,
    ):
        blocks = [nn.Linear(in_features, in_features)]

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            blocks.append(ResBlock(before, **kwargs))

            if before != after:
                blocks.append(nn.Linear(before, after))

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features


class NRE(nn.Module):
    r"""Neural Ratio Estimator (NRE)

    (theta, x) ---> log r(theta, x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the observations.
        moments: The parameters moments (mu, sigma) for standardization.
        arch: The network architecture (`MLP` or `ResMLP`).

        **kwargs are passed to `MLP` or `ResMLP`.

    References:
        [1] Likelihood-free MCMC with Amortized Approximate Ratio Estimators
        (Hermans et al., 2019)
        https://arxiv.org/abs/1903.04057
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        moments: Tuple[Tensor, Tensor] = None,
        arch: str = 'MLP',
        **kwargs,
    ):
        super().__init__()

        if moments is not None:
            mu, sigma = moments

        self.standardize = nn.Identity() if moments is None else Affine(-mu / sigma, 1 / sigma)
        self.broadcast = Broadcast(keep=1)

        if arch == 'ResMLP':
            arch = ResMLP
        else:  # arch == 'MLP'
            arch = MLP

        self.net = arch(theta_size + x_size, 1, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        theta = self.standardize(theta)
        theta, x = self.broadcast(theta, x)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class MNRE(nn.Module):
    r"""Marginal Neural Ratio Estimator (MNRE)

                ---> log r(theta_a, x)
               /
    (theta, x) ----> log r(theta_b, x)
               \
                ---> log r(theta_c, x)

    Args:
        masks: The masks of the considered parameter subspaces.
        x_size: The size of the observations.
        moments: The parameters moments (mu, sigma) for standardization.

        **kwargs are passed to `NRE`.
    """

    BASE = NRE

    def __init__(
        self,
        masks: BoolTensor,
        x_size: int,
        moments: Tuple[Tensor, Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer('masks', masks)

        if moments is not None:
            mu, sigma = moments

        self.estimators = nn.ModuleList([
            self.BASE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (mu[m], sigma[m]),
                **kwargs,
            ) for m in self.masks
        ])

    def __getitem__(self, mask: BoolTensor) -> nn.Module:
        r"""Select estimator r(theta_a, x)"""

        mask = mask.to(self.masks)
        select = torch.all(mask == self.masks, dim=-1)
        indices = torch.nonzero(select).squeeze(-1).tolist()

        for i in indices:
            return self.estimators[i]

        return None

    def filter(self, masks: Tensor):
        r"""Filter estimators within subspace"""

        estimators = []

        for m in masks:
            estimators.append(self[m])

        self.masks = masks
        self.estimators = nn.ModuleList(estimators)

    def forward(
        self,
        theta: Tensor,  # (N, D)
        x: Tensor,  # (N, L)
    ) -> Tensor:
        preds = []

        for mask, estimator in zip(self.masks, self.estimators):
            preds.append(estimator(theta[..., mask], x))

        return torch.stack(preds, dim=-1)


class AMNRE(NRE):
    r"""Arbitrary Marginal Neural Ratio Estimator (AMNRE)

    (theta, x, mask_a) ---> log r(theta_a, x)

    Args:
        theta_size: The size of the parameters.

        *args and **kwargs are passed to `NRE`.

    References:
        [1] Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference
        (Rozet et al., 2019)
        https://arxiv.org/abs/2110.00449
    """

    def __init__(
        self,
        theta_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_size * 2, *args, **kwargs)

        self.register_buffer('default', torch.ones(theta_size).bool())

    def __getitem__(self, mask: BoolTensor) -> nn.Module:
        r"""Select estimator r(theta_a, x)"""

        self.default = mask.to(self.default)

        return self

    def forward(
        self,
        theta: Tensor,  # (N, D)
        x: Tensor,  # (N, L)
        mask: BoolTensor = None,  # (D,) or (N, D)
    ) -> Tensor:
        if mask is None:
            mask = self.default

        zeros = theta.new_zeros(theta.shape[:-1] + mask.shape[-1:])

        if mask.dim() == 1 and theta.shape[-1] < mask.numel():
            theta = zeros.masked_scatter(mask, theta)
        else:
            theta = torch.where(mask, theta, zeros)

        theta = self.standardize(theta) * mask
        theta = torch.cat(self.broadcast(theta, mask * 2. - 1.), dim=-1)
        theta, x = self.broadcast(theta, x)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class NPE(nn.Module):
    r"""Neural Posterior Estimator (NPE)

    (theta, x) ---> log p(theta | x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the observations.
        moments: The parameters moments (mu, sigma) for standardization.

        **kwargs are passed to `MAF`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        moments: Tuple[Tensor, Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.broadcast = Broadcast(keep=1)
        self.flow = MAF(theta_size, x_size, moments=moments, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r""" log p(theta | x) """

        theta, x = self.broadcast(theta, x)

        return self.flow.log_prob(theta, x)

    def sample(self, x: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" theta ~ p(theta | x) """

        return self.flow.sample(x, shape)


class MNPE(MNRE):
    r"""Marginal Neural Posterior Estimator (MNPE)

                ---> log p(theta_a | x)
               /
    (theta, x) ----> log p(theta_b | x)
               \
                ---> log p(theta_c | x)

    Args:
        masks: The masks of the considered parameter subspaces.
        x_size: The size of the observations.
        moments: The parameters moments (mu, sigma) for standardization.

        **kwargs are passed to `NPE`.
    """

    BASE = NPE


class AMNPE(NPE):
    r"""Arbitrary Marginal Neural Posterior Estimator (AMNPE)

    (theta, x, mask_a) ---> log p(theta_a | x) / p(theta_a)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the observations.
        prior: The prior distributions p(theta).

        *args and **kwargs are passed to `NPE`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        prior: Distribution,
        *args,
        **kwargs,
    ):
        super().__init__(theta_size, x_size + theta_size, *args, **kwargs)

        self.prior = prior

        self.register_buffer('default', torch.ones(theta_size).bool())

    def __getitem__(self, mask: BoolTensor) -> nn.Module:
        r"""Select estimator p(theta_a | x)"""

        self.default = mask.to(self.default)

        return self

    def forward(
        self,
        theta: Tensor,  # (N, D)
        x: Tensor,  # (N, L)
        mask: BoolTensor = None,  # (D,) or (N, D)
    ) -> Tensor:
        if mask is None:
            mask = self.default

        theta_prime = self.prior.sample(theta.shape[:-1])

        if mask.dim() == 1 and theta.shape[-1] < mask.numel():
            theta = theta_prime.masked_scatter(mask, theta)
        else:
            theta = torch.where(mask, theta, theta_prime)

        x = torch.cat(self.broadcast(x, mask * 2. - 1.), dim=-1)
        theta, x = self.broadcast(theta, x)

        return self.flow.log_prob(theta, x) - self.prior.log_prob(theta)

    def sample(
        self,
        x: Tensor,  # (N, L)
        shape: torch.Size = (),
        mask: BoolTensor = None,  # (D,)
    ) -> Tensor:
        if mask is None:
            mask = self.default

        x = torch.cat(self.broadcast(x, mask * 2. - 1.), dim=-1)

        return self.flow.sample(x, shape)[..., mask]
