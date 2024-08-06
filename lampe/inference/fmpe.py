r"""Flow matching posterior estimation (FMPE) components.

The principle of FMPE is to train a regression network :math:`v_\phi(\theta, x, t)` to
approximate a vector field inducing a time-continuous normalizing flow between the
posterior distribution :math:`p(\theta | x)` and a standard Gaussian distribution
:math:`\mathcal{N}(0, I)`.

After training, the normalizing flow :math:`p_\phi(\theta | x)` induced by
:math:`v_\phi(\theta, x, t)` is used to evaluate the posterior density or generate
samples.

References:
    | Flow Matching for Generative Modeling (Lipman et al., 2023)
    | https://arxiv.org/abs/2210.02747

    | Flow Matching for Scalable Simulation-Based Inference (Dax et al., 2023)
    | https://arxiv.org/abs/2305.17161
"""

__all__ = [
    "FMPE",
    "FMPELoss",
]

import math
import torch
import torch.nn as nn

from torch import Tensor
from torch.distributions import Distribution
from typing import Callable
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

# isort: split
from ..nn import MLP


class FMPE(nn.Module):
    r"""Creates a flow matching posterior estimation (FMPE) network.

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        freqs: The number of time embedding frequencies.
        build: The network constructor (e.g. :class:`lampe.nn.ResMLP`). It takes the
            number of input and output features as positional arguments.
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        freqs: int = 3,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.net = build(theta_dim + x_dim + 2 * freqs, theta_dim, **kwargs)

        self.register_buffer("freqs", torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer("zeros", torch.zeros(theta_dim))
        self.register_buffer("ones", torch.ones(theta_dim))

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            t: The time :math:`t`, with shape :math:`(*,).`

        Returns:
            The vector field :math:`v_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        theta, x, t = broadcast(theta, x, t, ignore=1)

        return self.net(torch.cat((theta, x, t), dim=-1))

    def flow(self, x: Tensor) -> Distribution:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The normalizing flow :math:`p_\phi(\theta | x)`.
        """

        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self(theta, x, t),
                t0=x.new_tensor(0.0),
                t1=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )


class FMPELoss(nn.Module):
    r"""Creates a module that calculates the flow matching loss for a FMPE regressor.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N \|
            v_\phi((1 - t_i) \theta_i + (t_i + \eta) \epsilon_i, x_i, t_i)
            - (\epsilon_i - \theta_i)
        \|_2^2

    where :math:`t_i \sim \mathcal{U}(0, 1)` and :math:`\epsilon_i \sim \mathcal{N}(0, I)`.

    Arguments:
        estimator: A regression network :math:`v_\phi(\theta, x, t)`.
    """

    def __init__(self, estimator: nn.Module, eta: float = 1e-3):
        super().__init__()

        self.estimator = estimator
        self.eta = eta

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        t = torch.rand(theta.shape[:-1], dtype=theta.dtype, device=theta.device)
        t_ = t[..., None]

        eps = torch.randn_like(theta)
        theta_prime = (1 - t_) * theta + (t_ + self.eta) * eps
        v = eps - theta

        return (self.estimator(theta_prime, x, t) - v).square().mean()
