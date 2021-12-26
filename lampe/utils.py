r"""Miscellaneous tools and general purpose helpers"""

import os
import warnings

from functools import lru_cache, partial
from numpy import vectorize as _vectorize

from typing import Callable


def identity(f: Callable = None, *args, **kwargs) -> Callable:
    r"""Identity decorator"""

    if f is None:
        return lambda f: f
    else:
        return f


def cache(f: Callable = None, **kwargs) -> Callable:
    r"""Memory (RAM) cache decorator"""

    kwargs.setdefault('maxsize', None)

    if f is None:
        return lru_cache(**kwargs)

    return lru_cache(**kwargs)(f)


def disk_cache(*args, **kwargs) -> Callable:
    r"""Disk cache decorator"""

    try:
        from joblib import Memory
    except ImportError as e:
        warnings.warn(f'{e}. Disabling disk cache.', ImportWarning)
        return cache(*args, **kwargs)

    mem = Memory(os.path.expanduser('~/.cache'), mmap_mode='c', verbose=0)
    return mem.cache(*args, **kwargs)


def vectorize(f: Callable = None, **kwargs) -> Callable:
    r"""Convenience vectorization decorator"""

    if f is None:
        return partial(_vectorize, **kwargs)

    return _vectorize(f, **kwargs)


def jit(*args, **kwargs) -> Callable:
    r"""Just-in-Time (JIT) compilation decorator"""

    try:
        from numba import jit as njit
    except ImportError as e:
        warnings.warn(f'{e}. Disabling Just-in-Time compilation.', ImportWarning)
        return identity(*args, **kwargs)

    kwargs.setdefault('nopython', True)
    kwargs.setdefault('cache', True)

    return njit(*args, **kwargs)

