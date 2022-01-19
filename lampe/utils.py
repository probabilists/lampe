r"""Miscellaneous tools and general purpose helpers"""

import numpy as np
import os
import torch

from functools import lru_cache, partial, wraps
from typing import *


def decorator(decoration: Callable) -> Callable:
    r"""Wraps a decoration inside a decorator"""

    @wraps(decoration)
    def decorate(f: Callable = None, /, **kwargs) -> Callable:
        if f is None:
            return decoration(**kwargs)
        else:
            return decoration(**kwargs)(f)

    return decorate


@decorator
def cache(disk: bool = False, maxsize: int = None, **kwargs) -> Callable:
    r"""Caching decorator"""

    if disk:
        try:
            from joblib import Memory
        except ImportError as e:
            print(f'ImportWarning: {e}. Fallback to regular cache.')
        else:
            memory = Memory(os.path.expanduser('~/.cache'), mmap_mode='c', verbose=0)
            return partial(memory.cache, **kwargs)

    return lru_cache(maxsize=maxsize, **kwargs)


@decorator
def vectorize(**kwargs) -> Callable:
    r"""Vectorization decorator"""

    return partial(np.vectorize, **kwargs)


def deepapply(obj: Any, fn: Callable) -> Any:
    r"""Applies `fn` to all tensors referenced in `obj`"""

    if torch.is_tensor(obj):
        obj = fn(obj)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = deepapply(value, fn)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = deepapply(value, fn)
    elif isinstance(obj, tuple):
        obj = tuple(
            deepapply(value, fn)
            for value in obj
        )
    elif hasattr(obj, '__dict__'):
        deepapply(obj.__dict__, fn)

    return obj
