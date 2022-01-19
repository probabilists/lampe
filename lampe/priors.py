r"""Priors and distributions"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import *
from torch.distributions.constraints import interval
from typing import *

from .utils import deepapply


__init__ = Distribution.__init__

def init(self, *args, **kwargs):
    __init__(self, *args, **kwargs)

    self.__class__ = type(
        self.__class__.__name__,
        (self.__class__, nn.Module),
        {},
    )

    nn.Module.__init__(self)

Distribution.__init__ = init
Distribution._apply = deepapply
Distribution._validate_args = False


class Joint(Distribution):
    r"""Joint distribution of independent random variables"""

    arg_constraints: dict = {}

    def __init__(self, marginals: List[Distribution]):
        super().__init__()

        self.marginals = marginals

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([sum(
            dist.event_shape.numel()
            for dist in self.marginals
        )])

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size()

    def rsample(self, shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.rsample(shape)
            y = y.view(shape + (-1,))
            x.append(y)

        return torch.cat(x, dim=-1)

    def log_prob(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1]
        i, lp = 0, 0

        for dist in self.marginals:
            j = i + dist.event_shape.numel()
            y = x[..., i:j].view(shape + dist.event_shape)
            lp = lp + dist.log_prob(y)
            i = j

        return lp


class JointNormal(Distribution):
    r"""Joint distribution of independent normal random variables"""

    def __new__(cls, loc: Tensor, scale: Tensor, ndims: int = 1):
        return Independent(Normal(loc, scale), ndims)


class JointUniform(Distribution):
    r"""Joint distribution of independent uniform random variables"""

    def __new__(cls, low: Tensor, high: Tensor, ndims: int = 1):
        return Independent(Uniform(low, high), ndims)


class Sort(Distribution):
    r"""Sort of independent scalar random variables"""

    arg_constraints: dict = {}

    def __init__(
        self,
        base: Distribution,
        n: int = 2,
        descending: bool = False,
    ):
        super().__init__()

        assert len(base.event_shape) < 1, 'base must be scalar'

        self.base = base
        self.n = n
        self.descending = descending
        self.log_fact = math.log(math.factorial(n))

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([self.n])

    @property
    def batch_shape(self) -> torch.Size:
        return self.base.batch_shape

    def rsample(self, shape: torch.Size = ()) -> Tensor:
        value = torch.stack([
            self.base.rsample(shape)
            for _ in range(self.n)
        ], dim=-1)
        value, _ = torch.sort(value, dim=-1, descending=self.descending)

        return value

    def log_prob(self, value: Tensor) -> Tensor:
        if self.descending:
            ordered = value[..., :-1] >= value[..., 1:]
        else:
            ordered = value[..., :-1] <= value[..., 1:]

        ordered = ordered.all(dim=-1)

        return (
            ordered.log() +
            self.log_fact +
            self.base.log_prob(value).sum(dim=-1)
        )


class TopK(Sort):
    r"""Top k of independent scalar random variables"""

    def __init__(
        self,
        base: Distribution,
        k: int = 1,
        n: int = 2,
        **kwargs,
    ):
        assert base.cdf(base.sample()) >= 0, 'base must implement cdf'
        assert 1 <= k <= n, 'k should be in [1, n]'

        super().__init__(base, n, **kwargs)

        self.k = k
        self.log_fact = self.log_fact - math.log(math.factorial(n - k))

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size([self.k])

    def rsample(self, shape: torch.Size = ()) -> Tensor:
        return super().rsample(shape)[..., :self.k]

    def log_prob(self, value: Tensor) -> Tensor:
        cdf = self.base.cdf(value[..., -1])

        if not self.descending:
            cdf = 1 - cdf

        return (
            (self.n - self.k) * cdf.log() +
            super().log_prob(value)
        )


class Maximum(TopK):
    r"""Maximum of independent scalar random variables"""

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, 1, n)

        self.descending = True

    @property
    def event_shape(self) -> torch.Size:
        return torch.Size()

    def rsample(self, shape: torch.Size = ()) -> Tensor:
        return super().rsample(shape).squeeze(dim=-1)

    def log_prob(self, value: Tensor) -> Tensor:
        return super().log_prob(value.unsqueeze(dim=-1))


class Minimum(Maximum):
    r"""Minimum of independent scalar random variables"""

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__(base, n)

        self.descending = False


class TransformedUniform(TransformedDistribution):
    r"""T-uniform distribution"""

    def __init__(self, low: Tensor, high: Tensor, t: Transform):
        super().__init__(Uniform(t(low), t(high)), [t.inv])


class PowerUniform(TransformedUniform):
    r"""Power-uniform distribution"""

    def __init__(self, low: Tensor, high: Tensor, exponent: float):
        super().__init__(low, high, PowerTransform(exponent))


class CosineUniform(TransformedUniform):
    r"""Cosine-uniform distribution"""

    def __init__(self, low: Tensor, high: Tensor):
        super().__init__(low, high, CosineTransform())


class SineUniform(TransformedUniform):
    r"""Sine-uniform distribution"""

    def __init__(self, low: Tensor, high: Tensor):
        super().__init__(low, high, SineTransform())


class CosineTransform(Transform):
    r"""Transform via the mapping y = -cos(x)"""

    domain = interval(0, math.pi)
    codomain = interval(-1, 1)
    bijective = True

    def _call(self, x):
        return -x.cos()

    def _inverse(self, y):
        return (-y).acos()

    def log_abs_det_jacobian(self, x, y):
        return x.sin().abs().log()


class SineTransform(Transform):
    r"""Transform via the mapping y = sin(x)"""

    domain = interval(-math.pi / 2, math.pi / 2)
    codomain = interval(-1, 1)
    bijective = True

    def _call(self, x):
        return x.sin()

    def _inverse(self, y):
        return y.asin()

    def log_abs_det_jacobian(self, x, y):
        return x.cos().abs().log()
