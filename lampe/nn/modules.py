r"""Modules and layers"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor

from .flows import MAF


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


class BatchNorm(nn.BatchNorm1d):
    r"""Batch Normalization (BatchNorm) layer
    normalizing only the last dimension.
    """

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]

        x = x.view(-1, x.size(-1))
        x = super().forward(x)
        x = x.view(shape + x.shape[-1:])

        return x


ACTIVATIONS = {
    'ReLU': nn.ReLU,
    'PReLU': nn.PReLU,
    'ELU': nn.ELU,
    'CELU': nn.CELU,
    'SELU': nn.SELU,
    'GELU': nn.GELU,
}


NORMALIZATIONS = {
    'batch': BatchNorm,
    'group': nn.GroupNorm,
    'layer': nn.LayerNorm,
}


class MLP(nn.Sequential):
    r"""Multi-Layer Perceptron (MLP)

    Args:
        input_size: The input size.
        output_size: The output size.
        hidden_sizes: The (list of) intermediate sizes.
        bias: Whether to use bias or not.
        activation: The activation layer type.
        dropout: The dropout rate.
        normalization: The normalization layer type.
        activation_first: Whether the first layer is an activation or not.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] = [64, 64],
        bias: bool = True,
        activation: str = 'ReLU',
        dropout: float = 0.,
        normalization: str = None,
        activation_first: bool = False,
        **absorb,
    ):
        activation = ACTIVATIONS[activation]

        if dropout == 0.:
            dropout = lambda: None
        else:
            dropout = lambda: nn.Dropout(dropout)

        normalization = NORMALIZATIONS.get(normalization, lambda x: None)

        layers = [normalization(input_size), activation()] if activation_first else []

        for current_size, next_size in zip([input_size] + hidden_sizes, hidden_sizes + [output_size]):
            layers.extend([
                nn.Linear(current_size, next_size, bias),
                normalization(next_size),
                activation(),
                dropout(),
            ])

        layers = layers[:-3]
        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.input_size = input_size
        self.output_size = output_size


class ResBlock(MLP):
    r"""Residual Block (ResBlock)

    Args:
        size: The input, output and hidden sizes.

        **kwargs are passed to `MLP`.
    """

    def __init__(self, size: int, **kwargs):
        super().__init__(size, size, [size], activation_first=True, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        return input + super().forward(input)


class ResNet(nn.Sequential):
    r"""Residual Network (ResNet)

    Args:
        input_size: The input size.
        output_size: The output size.
        residual_sizes: The (list of) intermediate sizes.

        **kwargs are passed to `ResBlock`.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        residual_sizes: list[int] = [64, 64, 64],
        **kwargs,
    ):
        bias = kwargs.get('bias', True)

        for current_size, next_size in zip([input_size] + hidden_sizes, hidden_sizes + [output_size]):
            if current_size != next_size:
                blocks.append(nn.Linear(current_size, next_size, bias))

            blocks.append(ResBlock(next_size, **kwargs))

        blocks = blocks[:-1]

        super().__init__(*blocks)

        self.input_size = input_size
        self.output_size = output_size


class NRE(nn.Module):
    r"""Neural Ratio Estimator (NRE)

    (theta, x) ---> log r(theta, x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the (encoded) observations.
        moments: The parameters moments (mu, sigma) for standardization.
        embedding: An optional embedding for the observations.
        arch: The network architecture (`MLP` or `ResNet`).

        **kwargs are passed to `MLP` or `ResNet`.

    References:
        [1] Likelihood-free MCMC with Amortized Approximate Ratio Estimators
        (Hermans et al., 2019)
        https://arxiv.org/abs/1903.04057
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        moments: tuple[Tensor, Tensor] = None,
        embedding: nn.Module = nn.Identity(),
        arch: str = 'MLP',
        **kwargs,
    ):
        super().__init__()

        if arch == 'ResNet':
            arch = ResNet
        else:  # arch == 'MLP'
            arch = MLP

        self.net = arch(theta_size + x_size, 1, **kwargs)

        if moments is not None:
            mu, sigma = moments

        self.standardize = nn.Identity() if moments is None else Affine(-mu / sigma, 1 / sigma)
        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        return self.net(torch.cat([self.standardize(theta), x], dim=-1)).squeeze(-1)


class MNRE(nn.Module):
    r"""Marginal Neural Ratio Estimator (MNRE)

                ---> log r(theta_a, x)
               /
    (theta, x) ----> log r(theta_b, x)
               \
                ---> log r(theta_c, x)

    Args:
        masks: The masks of the considered parameter subspaces.
        x_size: The size of the (encoded) observations.
        moments: The parameters moments (mu, sigma) for standardization.
        embedding: An optional embedding for the observations.

        **kwargs are passed to `NRE`.
    """

    def __init__(
        self,
        masks: BoolTensor,
        x_size: int,
        moments: tuple[Tensor, Tensor] = None,
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__()

        self.register_buffer('masks', masks)

        if moments is not None:
            mu, sigma = moments

        self.estimators = nn.ModuleList([
            NRE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (mu[m], sigma[m]),
                **kwargs,
            ) for m in self.masks
        ])

        self.embedding = embedding

    def __getitem__(self, mask: BoolTensor) -> NRE:
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
        theta: Tensor,
        x: Tensor,
    ) -> Tensor:
        preds = []

        for mask, estimator in zip(self.masks, self.estimators):
            preds.append(estimator(theta[..., mask], x))

        return torch.stack(preds, dim=-1)


class AMNRE(nn.Module):
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
        super().__init__()

        self.net = NRE(theta_size * 2, *args, **kwargs)

        self.standardize, self.net.standardize = self.net.standardize, nn.Identity()
        self.embedding, self.net.embedding = self.net.embedding, nn.Identity()

        self.register_buffer('default', torch.ones(theta_size).bool())

    def __getitem__(self, mask: BoolTensor) -> nn.Module:
        r"""Select estimator r(theta_a, x)"""

        self.default = mask.to(self.default)

        return self

    def forward(
        self,
        theta: Tensor,  # (N, D)
        x: Tensor,  # (N, *)
        mask: BoolTensor = None,  # (*, D)
    ) -> Tensor:
        if mask is None:
            mask = self.default

        if mask.dim() == 1 and theta.size(-1) < mask.numel():
            blank = theta.new_zeros(theta.shape[:-1] + mask.shape)
            blank[..., mask] = theta
            theta = blank
        elif mask.dim() > 1 and theta.shape != mask.shape:
            batch_shape = theta.shape[:-1]
            view_shape = batch_shape + (1,) * (mask.dim() - 1)
            expand_shape = batch_shape + mask.shape[:-1]

            theta = theta.view(view_shape + theta.shape[-1:]).expand(expand_shape + theta.shape[-1:])
            x = x.view(view_shape + x.shape[-1:]).expand(expand_shape + x.shape[-1:])

        theta = self.standardize(theta) * mask
        theta = torch.cat(torch.broadcast_tensors(theta, mask * 2. - 1.), dim=-1)

        return super().net(theta, x)


class NPE(nn.Module):
    r"""Neural Posterior Estimator (NPE)

    (theta, x) ---> log p(theta | x)

    Args:
        theta_size: The size of the parameters.
        x_size: The size of the (encoded) observations.
        moments: The parameters moments (mu, sigma) for standardization.
        embedding: An optional embedding for the observations.

        **kwargs are passed to `MAF`.
    """

    def __init__(
        self,
        theta_size: int,
        x_size: int,
        moments: tuple[Tensor, Tensor] = None,
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__()

        self.flow = MAF(theta_size, x_size, moments=moments, **kwargs)

        self.embedding = embedding

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r""" log p(theta | x) """

        return self.flow.condition(x).log_prob(theta)

    def sample(self, x: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" theta ~ p(theta | x) """

        return self.flow.condition(x).sample(shape)


class MNPE(MNRE):
    r"""Marginal Neural Posterior Estimator (MNPE)

                ---> log p(theta_a | x)
               /
    (theta, x) ----> log p(theta_b | x)
               \
                ---> log p(theta_c | x)

    Args:
        masks: The masks of the considered parameter subspaces.
        x_size: The size of the (encoded) observations.
        moments: The parameters moments (mu, sigma) for standardization.
        embedding: An optional embedding for the observations.

        **kwargs are passed to `NPE`.
    """

    def __init__(
        self,
        masks: BoolTensor,
        x_size: int,
        moments: tuple[Tensor, Tensor] = None,
        embedding: nn.Module = nn.Identity(),
        **kwargs,
    ):
        super().__init__(masks, x_size, embedding=embedding)

        if moments is not None:
            mu, sigma = moments

        self.etimators = nn.ModuleList([
            NPE(
                m.sum().item(),
                x_size,
                moments=None if moments is None else (mu[m], sigma[m]),
                **kwargs,
            )
            for m in self.masks
        ])

    def __getitem__(self, mask: BoolTensor) -> NPE:
        r"""Select estimator p(theta_a | x)"""

        return super().__getitem__(mask)
