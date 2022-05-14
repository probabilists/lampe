r"""Neural networks, layers and modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from typing import *


__all__ = ['MLP', 'ResBlock', 'ResMLP']


class Affine(nn.Module):
    r"""Creates an element-wise affine layer.

    .. math:: y = x \times \alpha + \beta

    Arguments:
        shift: The shift term :math:`\beta`.
        scale: The scale factor :math:`\alpha`.
    """

    def __init__(self, shift: Tensor, scale: Tensor):
        super().__init__()

        self.shift = nn.Parameter(torch.tensor(shift).float())
        self.scale = nn.Parameter(torch.tensor(scale).float())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift


class BatchNorm0d(nn.BatchNorm1d):
    r"""Creates a batch normalization (BatchNorm) layer for scalars.

    References:
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        (Ioffe et al., 2015)
        https://arxiv.org/abs/1502.03167

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.BatchNorm1d`.
        kwargs: Keyword arguments passed to :class:`torch.nn.BatchNorm1d`.
    """

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape

        x = x.reshape(-1, shape[-1])
        x = super().forward(x)
        x = x.reshape(shape)

        return x


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Also known as fully connected feedforward network, an MLP is a series of
    non-linear parametric transformations

    .. math:: h_{i + 1} = a_{i + 1}(W_{i + 1}^T h_i + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and ouput feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an MLP
    are its weights and biases :math:`\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Wikipedia:
        https://en.wikipedia.org/wiki/Feedforward_neural_network

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        activation: The activation layer type.
        batchnorm: Whether to use batch normalization or not.
        dropout: The dropout rate.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.

    Example:
        >>> net = MLP(64, 1, [32, 16], activation='ELU')
        >>> net
        MLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
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
        activation = {
            'ReLU': nn.ReLU,
            'ELU': nn.ELU,
            'CELU': nn.CELU,
            'SELU': nn.SELU,
            'GELU': nn.GELU,
        }.get(activation, nn.ReLU)

        batchnorm = BatchNorm0d if batchnorm else lambda _: None
        dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers = []

        for before, after in zip(
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
    r"""Creates a residual block.

    A residual block is a function of the type

    .. math:: y = x + f(x),

    where :math:`f` is a non-linear parametric transformation. An MLP with a
    constant number of features in hidden layers is commonly used as :math:`f`.

    Arguments:
        features: The input, output and hidden features.
        hidden_layers: The number of hidden layers.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> net = ResBlock(32, hidden_layers=3, activation='ELU')
        >>> net
        ResBlock(
          (0): Linear(in_features=32, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ELU(alpha=1.0)
          (6): Linear(in_features=32, out_features=32, bias=True)
        )
    """

    def __init__(self, features: int, hidden_layers: int = 2, **kwargs):
        super().__init__(
            features,
            features,
            [features] * hidden_layers,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    Like the regular MLP, the ResMLP is a series of non-linear parametric
    transformations. However, it uses residual blocks as transformations, which
    reduces the vanishing of gradients and allows for deeper networks.

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        kwargs: Keyword arguments passed to :class:`ResBlock`.

    Example:
        >>> net = ResMLP(64, 1, [32, 16], activation='ELU')
        >>> net
        ResMLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ResBlock(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=32, out_features=32, bias=True)
          )
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ResBlock(
            (0): Linear(in_features=16, out_features=16, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=16, out_features=16, bias=True)
          )
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            if after != before:
                blocks.append(nn.Linear(before, after))

            blocks.append(ResBlock(after, **kwargs))

        blocks = blocks[:-1]

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features
