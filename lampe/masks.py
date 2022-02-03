r"""Masking helpers"""

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor, BoolTensor, LongTensor
from torch.distributions import Distribution
from typing import *


class MaskDistribution(Distribution):
    r"""Abstract mask distribution"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dummy = torch.tensor(0.)

    @property
    def device(self) -> torch.device:
        return self.dummy.device


class SelectionMask(Distribution):
    r"""Samples uniformly from a selection of masks"""

    def __init__(self, selection: BoolTensor):
        super().__init__(event_shape=selection.shape[-1:])

        self.selection = selection

    def rsample(self, shape: torch.Size = ()) -> BoolTensor:
        r""" a ~ p(a) """

        indices = torch.randint(len(self.selection), shape, device=self.device)
        return self.selection[indices]


class UniformMask(MaskDistribution):
    r"""Samples uniformly among all masks of size `size`"""

    def __init__(self, size: int):
        super().__init__(event_shape=(self.size,))

        self.size = size

    def rsample(self, shape: torch.Size = ()) -> BoolTensor:
        r""" a ~ p(a) """

        integers = torch.randint(1, 2 ** self.size, shape, device=self.device)
        return bit_repr(integers, self.size)


class PoissonMask(MaskDistribution):
    r"""Samples among all masks of size `size`,
    with the number of positive bits following a Poisson distribution"""

    def __init__(self, size: int, lmbda: float = 1.):
        super().__init__(event_shape=(self.size,))

        self.size = size
        self.lmbda = lmbda

        self.rng = np.random.default_rng()

    def rsample(self, shape: torch.Size = ()) -> BoolTensor:
        r""" a ~ p(a) """

        k = self.rng.poisson(self.lmbda, shape)
        k = torch.from_numpy(k).to(self.device)

        mask = torch.arange(self.size, device=self.device)
        mask = mask <= k[..., None]

        order = torch.rand(mask.shape, device=self.device)
        order = torch.argsort(order, dim=-1)

        return torch.gather(mask, dim=-1, index=order)


def str2mask(string: str) -> BoolTensor:
    return torch.tensor([char == '1' for char in string])


def mask2str(mask: BoolTensor) -> str:
    return ''.join('1' if bit else '0' for bit in mask)


def bit_repr(integers: LongTensor, bits: int) -> BoolTensor:
    r"""Bit representation of integers"""

    powers = 2 ** torch.arange(bits).to(integers)
    bits = integers[..., None].bitwise_and(powers) != 0

    return bits
