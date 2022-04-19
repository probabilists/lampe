r"""PyTorch monkey patches."""

import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Distribution
from torch.optim import Optimizer
from typing import *


################
# Distribution #
################

def new_init(self, *args, **kwargs):
    r"""Initializes :py:`self` with the features of a :class:`torch.nn.Module` instance."""

    old_init(self, *args, **kwargs)

    self.__class__ = type(
        self.__class__.__name__,
        (self.__class__, nn.Module),
        {},
    )

    nn.Module.__init__(self)

def deepapply(obj: Any, f: Callable) -> Any:
    r"""Applies :py:`f` to all tensors referenced in :py:`obj`."""

    if torch.is_tensor(obj):
        obj = f(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = deepapply(value, f)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = deepapply(value, f)
    elif isinstance(obj, tuple):
        obj = tuple(
            deepapply(value, f)
            for value in obj
        )
    elif hasattr(obj, '__dict__'):
        deepapply(obj.__dict__, f)

    return obj

old_init = Distribution.__init__
Distribution.__init__ = new_init
Distribution._apply = deepapply
Distribution._validate_args = False
Distribution.arg_constraints = {}


#############
# Optimizer #
#############

def lrs(self) -> Iterable[float]:
    r"""Yields the learning rates of the parameter groups."""

    return (group['lr'] for group in self.param_groups)

def parameters(self) -> Iterable[Tensor]:
    r"""Yields the parameter tensors of the parameter groups."""

    return (p for group in self.param_groups for p in group['params'])

Optimizer.lrs = lrs
Optimizer.parameters = parameters
