r"""Priors and distributions"""

import math
import torch

from torch.distributions import transforms, constraints
from torch.distributions import (
    Distribution,
    Independent,
    MultivariateNormal,
    Normal,
    TransformedDistribution,
    Uniform,
)

from torch import Tensor


Distribution.set_default_validate_args(False)


class Joint(Distribution):
    r"""Joint distribution of random variables"""

    arg_constraints: dict = {}

    def __init__(self, marginals: list[Distribution]):
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

    def sample(self, shape: torch.Size = ()):
        x = []

        for dist in self.marginals:
            y = dist.sample(shape)
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


class JointUniform(Distribution):
    r"""Joint distribution of uniform random variables"""

    def __new__(cls, low: Tensor, high: Tensor, ndims: int = 1):
        return Independent(Uniform(low, high), ndims)


class JointNormal(Distribution):
    r"""Joint distribution of normal random variables"""

    def __new__(cls, loc: Tensor, scale: Tensor, ndims: int = 1):
        return Independent(Normal(loc, scale), ndims)


class Sort(Distribution):
    r"""Sorted scalar random variables"""

    arg_constraints: dict = {}

    def __init__(self, base: Distribution, n: int = 2, descending: bool = False):
        super().__init__()

        self.base = base
        self.n = n
        self.descending = descending
        self.log_factor = math.log(math.factorial(n))

    @property
    def event_shape(self) -> torch.Size:
        return self.base.event_shape + (self.n,)

    def sample(self, shape: torch.Size = ()) -> Tensor:
        value = self.base.sample(shape + (self.n,))
        value, _ = torch.sort(value, dim=-1, descending=self.descending)

        return value

    def log_prob(self, value: Tensor) -> Tensor:
        if self.descending:
            ordered = value[..., :-1] >= value[..., 1:]
        else:
            ordered = value[..., :-1] <= value[..., 1:]

        ordered = ordered.all(axis=-1)

        return self.log_factor + ordered.log() + self.base.log_prob(value).sum(dim=-1)


class Maximum(Distribution):
    r"""Maximum of scalar random variables"""

    arg_constraints: dict = {}

    def __init__(self, base: Distribution, n: int = 2):
        super().__init__()

        self.n = n
        self.log_n = math.log(n)

    def sample(self, shape: torch.Size = ()) -> Tensor:
        value = self.base.sample(shape + (self.n,))
        value, _ = torch.max(value, dim=-1)

        return value

    def log_prob(self, value: Tensor) -> Tensor:
        log_cdf = self.base.cdf(value).log()
        return self.log_n + (self.n - 1) * log_cdf + self.base.log_prob(value)


class Minimum(Maximum):
    r"""Minimum of scalar random variables"""

    def sample(self, shape: torch.Size = ()) -> Tensor:
        value = self.base.sample(shape + (self.n,))
        value, _ = torch.min(value, dim=-1)

        return value

    def log_prob(self, value: Tensor) -> Tensor:
        log_cdf = (1 - self.base.cdf(value)).log()
        return self.log_n + (self.n - 1) * log_cdf + self.base.log_prob(value)


class SineTransform(transforms.Transform):
    r"""Transform via the mapping y = sin(x)"""

    domain = constraints.interval(-math.pi / 2, math.pi / 2)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def _call(self, x):
        return x.sin()

    def _inverse(self, y):
        return y.asin()

    def log_abs_det_jacobian(self, x, y):
        return x.cos().abs().log()


class CosineTransform(transforms.Transform):
    r"""Transform via the mapping y = -cos(x)"""

    domain = constraints.interval(0, math.pi)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def _call(self, x):
        return -x.cos()

    def _inverse(self, y):
        return (-y).acos()

    def log_abs_det_jacobian(self, x, y):
        return x.sin().abs().log()


class SineUniform(TransformedDistribution):
    r"""Sine-uniform distribution"""

    def __new__(cls, low: Tensor, high: Tensor):
        t = SineTransform()
        return TransformedDistribution(Uniform(t(low), t(high)), [t.inv])


class CosineUniform(TransformedDistribution):
    r"""Cosine-uniform distribution"""

    def __new__(cls, low: Tensor, high: Tensor):
        t = CosineTransform()
        return TransformedDistribution(Uniform(t(low), t(high)), [t.inv])


class PowerUniform(TransformedDistribution):
    r"""Power-uniform distribution"""

    def __new__(cls, low: Tensor, high: Tensor, exponent: float):
        t = transforms.PowerTransform(exponent)
        return TransformedDistribution(Uniform(t(low), t(high)), [t.inv])
