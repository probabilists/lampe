r"""Masking helpers."""

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor, BoolTensor
from torch.distributions import *
from typing import *


def mask2str(b: BoolTensor) -> str:
    r"""Represents a binary mask as a string.

    Arguments:
        b: A binary mask :math:`b`, with shape :math:`(D,)`.

    Example:
        >>> b = torch.tensor([True, True, False, True, False])
        >>> mask2str(b)
        '11010'
    """

    return ''.join('1' if bit else '0' for bit in b)


def str2mask(string: str) -> BoolTensor:
    r"""Parses the string representation of a binary mask into a tensor.

    Arguments:
        string: A binary mask string representation.

    Example:
        >>> str2mask('11010')
        tensor([True, True, False, True, False])
    """

    return torch.tensor([char == '1' for char in string])


class BernoulliMask(Independent):
    r"""Creates a distribution :math:`P(b)` over all binary masks :math:`b` in the
    hypercube :math:`\{0, 1\}^D` such that each bit :math:`b_i` has a probability
    :math:`p` of being positive.

    .. math:: P(b) = \prod^D_{i = 1} p^{b_i} (1 - p)^{1 - b_i}

    Arguments:
        dim: The hypercube dimensionality :math:`D`.
        p: The probability :math:`p` of a bit to be positive.

    Example:
        >>> d = BernoulliMask(5, 0.5)
        >>> d.sample()
        tensor([True, True, False, True, False])
    """

    has_rsample = False

    def __init__(self, dim: int, p: float = 0.5):
        super().__init__(Bernoulli(torch.ones(dim) * p), 1)

    def log_prob(b: BoolTensor) -> Tensor:
        return super().log_prob(b.float())

    def sample(self, shape: torch.Size = ()) -> BoolTensor:
        return super().sample(shape).bool()


class SelectionMask(Distribution):
    r"""Creates a mask distribution :math:`P(b)`, uniform over a selection of
    binary masks :math:`\mathcal{B} \subseteq \{0, 1\}^D`.

    .. math:: P(b) = \begin{cases}
            \frac{1}{|\mathcal{B}|} & \text{if } b \in \mathcal{B} \\
            0 & \text{otherwise}
        \end{cases}

    Arguments:
        selection: A binary mask selection :math:`\mathcal{B}`.

    Example:
        >>> selection = torch.tensor([
        ...     [True, False, False],
        ...     [False, True, False],
        ...     [False, False, True],
        ... ])
        >>> d = SelectionMask(selection)
        >>> d.sample()
        tensor([False, True, False])
    """

    def __init__(self, selection: BoolTensor):
        super().__init__(event_shape=selection.shape[-1:])

        self.selection = selection

    def log_prob(b: BoolTensor) -> Tensor:
        match = torch.all(b[..., None, :] == self.selection, dim=-1)
        prob = match.float().mean(dim=-1)
        return prob.log()

    def sample(self, shape: torch.Size = ()) -> BoolTensor:
        index = torch.randint(len(self.selection), shape, device=self.device)
        return self.selection[index]
