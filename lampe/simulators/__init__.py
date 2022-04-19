r"""Simulators and benchmarks."""

from abc import ABC, abstractmethod
from numpy import ndarray as Array
from torch import Tensor
from typing import *


class Simulator(ABC):
    r"""Abstract simulator class."""

    @abstractmethod
    def __call__(self, theta: Union[Array, Tensor]) -> Union[Array, Tensor]:
        pass
