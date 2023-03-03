r"""Neural networks, layers and modules."""

__all__ = ['MLP', 'ResMLP']

import torch
import torch.nn as nn

from torch import Tensor
from typing import *
from zuko.nn import MLP


class Residual(nn.Module):
    r"""Creates a residual block from a non-linear function :math:`f`.

    .. math:: y = x + f(x)

    Arguments:
        f: A function :math:`f`.
    """

    def __init__(self, f: nn.Module):
        super().__init__()

        self.f = f

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.f})'

    def forward(self, x: Tensor) -> Tensor:
        return x + self.f(x)


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    A ResMLP is a series of residual blocks where each block is a (shallow) MLP. Using
    residual blocks instead of regular non-linear functions prevents the gradients from
    vanishing, which allows for deeper networks.

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> net = ResMLP(64, 1, [32, 16], activation=nn.ELU)
        >>> net
        ResMLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): Residual(MLP(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=32, out_features=32, bias=True)
          ))
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): Residual(MLP(
            (0): Linear(in_features=16, out_features=16, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=16, out_features=16, bias=True)
          ))
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Sequence[int] = (64, 64),
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            (in_features, *hidden_features),
            (*hidden_features, out_features),
        ):
            if after != before:
                blocks.append(nn.Linear(before, after))

            blocks.append(Residual(MLP(after, after, [after], **kwargs)))

        blocks = blocks[:-1]

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features
