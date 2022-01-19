r"""Mask samplers and helpers"""

import numpy as np
import torch
import torch.nn as nn

from torch import Tensor, BoolTensor, LongTensor
from typing import *


class MaskSampler(nn.Module):
    r"""Abstract mask sampler"""

    def __init__(self, filtr: Union[BoolTensor, str] = None):
        super().__init__()

        if filtr is None:
            self.filtr = None
        else:
            if type(filtr) is str:
                filtr = str2mask(filtr)

            self.register_buffer('filtr', filtr)

        self.register_buffer('dummy', torch.tensor(0.))

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def sample(self, shape: Tuple[int] = ()) -> BoolTensor:
        r""" a ~ p(a) """

        mask = self.forward(shape)

        if self.filtr is not None:
            temp = mask.new_zeros(shape + self.filtr.shape)
            temp[..., self.filtr] = mask
            mask = temp

        return mask


def str2mask(string: str) -> BoolTensor:
    return torch.tensor([char == '1' for char in string])


def mask2str(mask: BoolTensor) -> str:
    return ''.join('1' if bit else '0' for bit in mask)


class SelectionMask(MaskSampler):
    r"""Samples uniformly from a selection of masks"""

    def __init__(self, selection: BoolTensor, **kwargs):
        super().__init__(**kwargs)

        self.register_buffer('selection', selection)

    def forward(self, shape: Tuple[int] = ()) -> BoolTensor:
        indices = torch.randint(len(self.selection), shape, device=self.device)
        return self.selection[indices]


class UniformMask(MaskSampler):
    r"""Samples uniformly among all masks of size `size`"""

    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)

        self.size = size

    def forward(self, shape: Tuple[int] = ()) -> BoolTensor:
        integers = torch.randint(1, 2 ** self.size, shape, device=self.device)
        return bit_repr(integers, self.size)


def bit_repr(integers: LongTensor, bits: int) -> BoolTensor:
    r"""Bit representation of integers"""

    powers = 2 ** torch.arange(bits).to(integers)
    bits = integers[..., None].bitwise_and(powers) != 0

    return bits


class PoissonMask(MaskSampler):
    r"""Samples among all masks of size `size`,
    with the number of positive bits following a Poisson distribution"""

    def __init__(self, size: int, lmbda: float = 1., **kwargs):
        super().__init__(**kwargs)

        self.size = size
        self.lmbda = lmbda

        self.rng = np.random.default_rng()

    def forward(self, shape: Tuple[int] = ()) -> BoolTensor:
        k = self.rng.poisson(self.lmbda, shape)
        k = torch.from_numpy(k).to(self.device)

        mask = torch.arange(self.size, device=self.device)
        mask = mask <= k[..., None]

        order = torch.rand(mask.shape, device=self.device)
        order = torch.argsort(order, dim=-1)

        return torch.gather(mask, dim=-1, index=order)
