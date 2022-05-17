r"""Parameterizable transformations and bijections."""

import math
import torch
import torch.nn.functional as F

from torch import Tensor, LongTensor
from torch.distributions import Transform, constraints
from typing import *

from ..utils import broadcast


torch.distributions.transforms._InverseTransform.__name__ = 'Inverse'


class CosTransform(Transform):
    r"""Transform via the mapping :math:`y = -\cos(x)`."""

    domain = constraints.interval(0, math.pi)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CosTransform)

    def _call(self, x: Tensor) -> Tensor:
        return -x.cos()

    def _inverse(self, y: Tensor) -> Tensor:
        return (-y).acos()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.sin().abs().log()


class SinTransform(Transform):
    r"""Transform via the mapping :math:`y = \sin(x)`."""

    domain = constraints.interval(-math.pi / 2, math.pi / 2)
    codomain = constraints.interval(-1, 1)
    bijective = True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SinTransform)

    def _call(self, x: Tensor) -> Tensor:
        return x.sin()

    def _inverse(self, y: Tensor) -> Tensor:
        return y.asin()

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.cos().abs().log()


class MonotonicAffineTransform(Transform):
    r"""Transform via the mapping :math:`y = x \times \text{softplus}(\alpha) + \beta`.

    Arguments:
        shift: The shift term :math:`\beta`, with shape :math:`(*,)`.
        scale: The unconstrained scale factor :math:`\alpha`, with shape :math:`(*,)`.
        eps: A numerical stability term.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        shift: Tensor,
        scale: Tensor,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.shift = shift
        self.scale = F.softplus(scale) + eps

    def _call(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

    def _inverse(self, y: Tensor) -> Tensor:
        return (y - self.shift) / self.scale

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.log(self.scale).expand(x.shape)


class MonotonicRationalQuadraticSplineTransform(Transform):
    r"""Transform via a monotonic rational-quadratic spline mapping.

    References:
        Neural Spline Flows (Durkan et al., 2019)
        https://arxiv.org/abs/1906.04032

    Arguments:
        widths: The unconstrained bin widths, with shape :math:`(*, K)`.
        heights: The unconstrained bin heights, with shape :math:`(*, K)`.
        derivatives: The unconstrained knot derivatives, with shape :math:`(*, K - 1)`.
        bound: The spline's (co)domain bound :math:`B`.
        eps: A numerical stability term.
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(
        self,
        widths: Tensor,
        heights: Tensor,
        derivatives: Tensor,
        bound: float = 3,
        eps: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        widths = F.softmax(widths, dim=-1) + eps
        heights = F.softmax(heights, dim=-1) + eps
        derivatives = F.softplus(derivatives) + eps

        self.horizontal = 2 * bound * torch.cumsum(F.pad(widths, (1, 0), value=-1), dim=-1)
        self.vertical = 2 * bound * torch.cumsum(F.pad(heights, (1, 0), value=-1), dim=-1)
        self.derivatives = F.pad(derivatives, (1, 1), value=1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(bins={self.bins})'

    @property
    def bins(self) -> int:
        return self.horizontal.shape[-1] - 1

    def bin(self, k: LongTensor) -> Tuple:
        mask = torch.logical_and(0 <= k, k < self.bins)

        k = k % self.bins
        k0_k1 = torch.stack((k, k+1))

        k0_k1, hs, vs, ds = broadcast(
            k0_k1[..., None],
            self.horizontal,
            self.vertical,
            self.derivatives,
            ignore=1,
        )

        x0, x1 = hs.gather(-1, k0_k1).squeeze(-1)
        y0, y1 = vs.gather(-1, k0_k1).squeeze(-1)
        d0, d1 = ds.gather(-1, k0_k1).squeeze(-1)

        s = (y1 - y0) / (x1 - x0)

        return mask, x0, x1, y0, y1, d0, d1, s

    @staticmethod
    def searchsorted(seq: Tensor, value: Tensor) -> LongTensor:
        return torch.sum(seq < value[..., None], dim=-1) - 1

    def _call(self, x: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        y = (
            y0 +
            (y1 - y0) *
            (s * z**2 + d0 * z * (1 - z)) /
            (s + (d0 + d1 - 2 * s) * z * (1 - z))
        )

        return torch.where(mask, y, x)

    def _inverse(self, y: Tensor) -> Tensor:
        k = self.searchsorted(self.vertical, y)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        y_ = mask * (y - y0)

        a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2 * s)
        b = (y1 - y0) * d0 - y_ * (d0 + d1 - 2 * s)
        c = -s * y_

        z = 2 * c / (-b - (b**2 - 4 * a * c).sqrt())

        x = x0 + z * (x1 - x0)

        return torch.where(mask, x, y)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        k = self.searchsorted(self.horizontal, x)
        mask, x0, x1, y0, y1, d0, d1, s = self.bin(k)

        z = mask * (x - x0) / (x1 - x0)

        jacobian = (
            s**2
            * (2 * s * z * (1 - z) + d0 * (1 - z)**2 + d1 * z**2)
            / (s + (d0 + d1 - 2 * s) * z * (1 - z))**2
        )

        return torch.log(jacobian) * mask


class LULinearTransform(Transform):
    r"""Transform via the mapping :math:`y = LU x`.

    The diagonal elements of :math:`L` and :math:`U` are set to 1.

    Arguments:
        lower: The lower-triangular elements of :math:`L`, with shape
            :math:`(*, \frac{D (D - 1)}{2})`.
        upper: The upper-triangular elements of :math:`U`, with shape
            :math:`(*, \frac{D (D - 1)}{2})`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        lower: Tensor,
        upper: Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        d = int((2 * lower.shape[-1])**(1 / 2) + 1)

        identity = torch.eye(d).reshape(-1).to(lower).expand(*lower.shape[:-1], -1)
        mask = lower.new_ones((d, d), dtype=bool)
        mask = torch.tril(mask, diagonal=-1)

        lower = torch.masked_scatter(identity, mask.reshape(-1), lower)
        upper = torch.masked_scatter(identity, mask.t().reshape(-1), upper)

        self.lower = lower.reshape(lower.shape[:-1] + (d, d))
        self.upper = upper.reshape(upper.shape[:-1] + (d, d))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.lower.shape[-1]})'

    def _call(self, x: Tensor) -> Tensor:
        return torch.matmul(self.lower, torch.matmul(self.upper, x[..., None])).squeeze(-1)

    def _inverse(self, y: Tensor) -> Tensor:
        return torch.triangular_solve(
            torch.triangular_solve(
                y[..., None],
                self.lower,
                upper=False,
                unitriangular=True,
            ).solution,
            self.upper,
            upper=True,
            unitriangular=True,
        ).solution.squeeze(-1)

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.new_zeros(x.shape[:-1])


class Permutation(Transform):
    r"""Transform via a permutation of the elements.

    Arguments:
        order: The permuatation order, with shape :math:`(*, D)`.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        order: LongTensor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.order = order

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.order.tolist()})'

    def _call(self, x: Tensor) -> Tensor:
        return x.gather(-1, self.order.expand(x.shape))

    def _inverse(self, y: Tensor) -> Tensor:
        return y.gather(-1, torch.argsort(self.order, -1).expand(y.shape))

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return x.new_zeros(x.shape[:-1])


class AutoregressiveTransform(Transform):
    r"""Tranform via an autoregressive mapping.

    Arguments:
        meta: A meta function :math:`g(x) = f`.
        passes: The number of passes for the inverse transformation.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True

    def __init__(
        self,
        meta: Callable[[Tensor], Transform],
        passes: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.meta = meta
        self.passes = passes

    def _call(self, x: Tensor) -> Tensor:
        return self.meta(x)(x)

    def _inverse(self, y: Tensor) -> Tensor:
        x = torch.zeros_like(y)
        for _ in range(self.passes):
            x = self.meta(x).inv(y)

        return x

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        return self.meta(x).log_abs_det_jacobian(x, y).sum(dim=-1)
