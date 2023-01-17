r"""Inference components such as estimators, training losses and MCMC samplers."""

__all__ = [
    'NRE',
    'NRELoss',
    'BNRELoss',
    'AMNRE',
    'AMNRELoss',
    'NPE',
    'NPELoss',
    'AMNPE',
    'AMNPELoss',
    'NSE',
    'NSELoss',
    'MetropolisHastings',
]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torch import Tensor, BoolTensor, Size
from typing import *

from zuko.distributions import Distribution, DiagNormal, NormalizingFlow
from zuko.flows import FlowModule, MAF
from zuko.transforms import FFJTransform
from zuko.utils import broadcast

from .nn import MLP


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) classifier network.

    The principle of neural ratio estimation is to train a classifier network
    :math:`d_\phi(\theta, x)` to discriminate between pairs :math:`(\theta, x)` equally
    sampled from the joint distribution :math:`p(\theta, x)` and the product of the
    marginals :math:`p(\theta)p(x)`. Formally, the optimization problem is

    .. math:: \arg\min_\phi
        \mathbb{E}_{p(\theta, x)} \big[ \ell(d_\phi(\theta, x)) \big] +
        \mathbb{E}_{p(\theta)p(x)} \big[ \ell(1 - d_\phi(\theta, x)) \big]

    where :math:`\ell(p) = -\log p` is the negative log-likelihood. For this task, the
    decision function modeling the Bayes optimal classifier is

    .. math:: d(\theta, x) = \frac{p(\theta, x)}{p(\theta, x) + p(\theta) p(x)}

    thereby defining the likelihood-to-evidence (LTE) ratio

    .. math:: r(\theta, x)
        = \frac{d(\theta, x)}{1 - d(\theta, x)}
        = \frac{p(\theta, x)}{p(\theta) p(x)}
        = \frac{p(x | \theta)}{p(x)}
        = \frac{p(\theta | x)}{p(\theta)} .

    To prevent numerical stability issues when :math:`d_\phi(\theta, x) \to 0`,
    the neural network returns the logit of the class prediction
    :math:`\text{logit}(d_\phi(\theta, x)) = \log r_\phi(\theta, x)`.

    References:
        | Approximating Likelihood Ratios with Calibrated Discriminative Classifiers (Cranmer et al., 2015)
        | https://arxiv.org/abs/1506.02169

        | Likelihood-free MCMC with Amortized Approximate Ratio Estimators (Hermans et al., 2019)
        | https://arxiv.org/abs/1903.04057

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        build: The network constructor (e.g. :class:`lampe.nn.ResMLP`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.net = build(theta_dim + x_dim, 1, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta, x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class NRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for a NRE classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class BNRELoss(nn.Module):
    r"""Creates a module that calculates the balanced cross-entropy loss for a balanced
    NRE (BNRE) classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{N} \sum_{i = 1}^N
            \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\theta_i, x_i) + d_\phi(\theta_{i+1}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    References:
        | Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation (Delaunoy et al., 2022)
        | https://arxiv.org/abs/2208.13624

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x)`.
        lmbda: The weight :math:`\lambda` controlling the strength of the balancing
            condition.
    """

    def __init__(self, estimator: nn.Module, lmbda: float = 100.0):
        super().__init__()

        self.estimator = estimator
        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()

        return l1 + l0 + self.lmbda * lb


class AMNRE(NRE):
    r"""Creates an arbitrary marginal neural ratio estimation (AMNRE) classifier
    network.

    The principle of AMNRE is to introduce, as input to the classifier, a binary mask
    :math:`b \in \{0, 1\}^D` indicating a subset of parameters :math:`\theta_b =
    (\theta_i: b_i = 1)` of interest. Intuitively, this allows the classifier to
    distinguish subspaces and to learn a different ratio for each of them. Formally, the
    classifier network takes the form :math:`d_\phi(\theta_b, x, b)` and the
    optimization problem becomes

    .. math:: \arg\min_\phi
        \mathbb{E}_{p(\theta, x) P(b)} \big[ \ell(d_\phi(\theta_b, x, b)) \big] +
        \mathbb{E}_{p(\theta)p(x) P(b)} \big[ \ell(1 - d_\phi(\theta_b, x, b)) \big],

    where :math:`P(b)` is a binary mask distribution. In this context, the Bayes
    optimal classifier is

    .. math:: d(\theta_b, x, b)
        = \frac{p(\theta_b, x)}{p(\theta_b, x) + p(\theta_b) p(x)}
        = \frac{r(\theta_b, x)}{1 + r(\theta_b, x)} .

    Therefore, a classifier network trained for AMNRE gives access to an estimator
    :math:`\log r_\phi(\theta_b, x, b)` of all marginal LTE log-ratios
    :math:`\log r(\theta_b, x)`.

    References:
        | Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference (Rozet et al., 2021)
        | https://arxiv.org/abs/2110.00449

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        args: Positional arguments passed to :class:`NRE`.
        kwargs: Keyword arguments passed to :class:`NRE`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim, x_dim + theta_dim, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`, or
                a subset :math:`\theta_b`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta_b, x, b)`, with shape :math:`(*,)`.
        """

        if theta.shape[-1] < b.shape[-1]:
            theta, b = broadcast(theta, b, ignore=1)
            theta = theta.new_zeros(b.shape).masked_scatter(b, theta)

        theta, x, b = broadcast(theta * b, x, b * 2.0 - 1.0, ignore=1)

        return self.net(torch.cat((theta, x, b), dim=-1)).squeeze(-1)


class AMNRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for an AMNRE classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i \odot b_i, x_i, b_i)) +
        \ell(1 - d_\phi(\theta_{i+1} \odot b_i, x_i, b_i))

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A classifier network :math:`d_\phi(\theta, x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
    ):
        super().__init__()

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        b = self.mask_dist.sample(theta.shape[:-1])

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
            b,
        )

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return l1 + l0


class NPE(nn.Module):
    r"""Creates a neural posterior estimation (NPE) normalizing flow.

    The principle of neural posterior estimation is to train a parametric conditional
    distribution :math:`p_\phi(\theta | x)` to approximate the posterior distribution
    :math:`p(\theta | x)`. The optimization problem is to minimize the expected
    Kullback-Leibler (KL) divergence between the two distributions for all observations
    :math:`x \sim p(x)`, that is,

    .. math::
        \arg\min_\phi & ~ \mathbb{E}_{p(x)}
            \Big[ \text{KL} \big( p(\theta|x) \parallel p_\phi(\theta | x) \big) \Big] \\
        = \arg\min_\phi & ~ \mathbb{E}_{p(x)} \, \mathbb{E}_{p(\theta | x)}
            \left[ \log \frac{p(\theta | x)}{p_\phi(\theta | x)} \right] \\
        = \arg\min_\phi & ~ \mathbb{E}_{p(\theta, x)}
            \big[ -\log p_\phi(\theta | x) \big] .

    Normalizing flows are typically used for :math:`p_\phi(\theta | x)` as they are
    differentiable parametric distributions enabling gradient-based optimization
    techniques.

    Wikipedia:
        https://wikipedia.org/wiki/Kullback-Leibler_divergence

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        build: The flow constructor (e.g. :class:`zuko.flows.NSF`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], FlowModule] = MAF,
        **kwargs,
    ):
        super().__init__()

        self.flow = build(theta_dim, x_dim, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.flow(x).log_prob(theta)

    def sample(self, x: Tensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            shape: The shape :math:`S` of the samples.

        Returns:
            The samples :math:`\theta \sim p_\phi(\theta | x)`,
            with shape :math:`S + (*, D)`.
        """

        with torch.no_grad():
            return self.flow(x).sample(shape)


class NPELoss(nn.Module):
    r"""Creates a module that calculates the negative log-likelihood loss for a NPE
    normalizing flow.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N -\log p_\phi(\theta_i | x_i) .

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_p = self.estimator(theta, x)

        return -log_p.mean()


class AMNPE(NPE):
    r"""Creates an arbitrary marginal neural posterior estimation (AMNPE)
    normalizing flow.

    TODO

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        args: Positional arguments passed to :class:`NPE`.
        kwargs: Keyword arguments passed to :class:`NPE`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim, x_dim + theta_dim, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x, b)`, with shape :math:`(*,)`.
        """

        theta, x, b = broadcast(theta, x, b * 2.0 - 1.0, ignore=1)

        return self.flow(torch.cat((x, b), dim=-1)).log_prob(theta)

    def sample(self, x: Tensor, b: BoolTensor, shape: Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(D,)`.
            shape: The shape :math:`S` of the samples.

        Returns:
            The samples :math:`\theta_b \sim p_\phi(\theta_b | x, b)`,
            with shape :math:`S + (*, D)`.
        """

        x, b_ = broadcast(x, b * 2.0 - 1.0, ignore=1)

        with torch.no_grad():
            return self.flow(torch.cat((x, b_), dim=-1)).sample(shape)[..., b]


class AMNPELoss(nn.Module):
    r"""Creates a module that calculates the negative log-likelihood loss for an AMNPE
    normalizing flow.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        -\log p_\phi(\theta_i \odot b_i + \theta_{i + 1} \odot (1 - b_i) | x_i, b_i)

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A normalizing flow :math:`p_\phi(\theta | x, b)`.
        mask_dist: A binary mask distribution :math:`P(b)`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        mask_dist: Distribution,
    ):
        super().__init__()

        self.estimator = estimator
        self.mask_dist = mask_dist

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        theta_prime = torch.roll(theta, 1, dims=0)

        b = self.mask_dist.sample(theta.shape[:-1])
        theta = torch.where(b, theta, theta_prime)

        log_prob = self.estimator(theta, x, b)

        return -log_prob.mean()


class NSE(nn.Module):
    r"""Creates a neural score estimation (NSE) regression network.

    The principle of neural score estimation is to train a regression network
    :math:`s_\phi(\theta_t, x, t)` to approximate the score :math:`\nabla_{\! \theta_t}
    \log p(\theta_t | x)` of the sub-variance preserving (sub-VP) diffusion process

    .. math:: \mathrm{d} \theta_t = -\frac{1}{2} \beta(t) \theta_t \, \mathrm{d} t +
        \sqrt{\beta(t) (1 - \alpha(t)^2)} \, \mathrm{d} w

    where :math:`\alpha(t) = \exp(-\int_0^t \beta(u) \, \mathrm{d} u)` and
    :math:`\beta(t) = (\beta_\max - \beta_\min) \, t + \beta_\min` . The optimization
    problem is to minimize the rescaled score-matching objective, that is,

    .. math:: \arg\min_\phi \mathbb{E}_{p(\theta, x) p(t) p(\theta_t | \theta)}
        \Big[ \big\| s_\phi(\theta_t, x, t) - (1 - \alpha(t)) \nabla_{\! \theta_t}
        \log p(\theta_t | \theta) \big\|_2^2 \Big]

    where :math:`p(\theta_t | \theta) = \mathcal{N}(\theta_t; \sqrt{\alpha(t)} \,
    \theta, (1 - \alpha(t))^2 I)` is the perturbation kernel corresponding to the
    diffusion process and for which the optimal regressor is the rescaled score

    .. math:: s(\theta_t, x, t) =
        (1 - \alpha(t)) \nabla_{\! \theta_t} \log p(\theta_t | x) .

    Given the latter, or an estimator of the latter, the probability flow ODE

    .. math:: \mathrm{d} \theta_t =
        \left[ -\frac{1}{2} \beta(t) \theta_t - \frac{1}{2} \beta(t) (1 + \alpha(t))
        s(\theta_t, x, t) \right] \, \mathrm{d} t

    shares the same marginal densities :math:`p(\theta_t | x)` as the diffusion process
    and can be efficiently integrated (forward for log-density computation and backward
    for sampling) with black-box ODE solvers.

    References:
        | Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
        | https://arxiv.org/abs/2011.13456

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        t_dim: The dimensionality of the time embedding.
        build: The network constructor (e.g. :class:`lampe.nn.ResMLP`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        t_dim: int = 1,
        build: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        self.net = build(theta_dim + x_dim + t_dim * 2, theta_dim, **kwargs)

        self.register_buffer('periods', torch.arange(t_dim) + 1)
        self.register_buffer('zeros', torch.zeros(theta_dim))
        self.register_buffer('ones', torch.ones(theta_dim))

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            t: The time :math:`t`, with shape :math:`(*,).`

        Returns:
            The rescaled score :math:`s_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        t = self.periods * math.pi * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        theta, x, t = broadcast(theta, x, t, ignore=1)

        return self.net(torch.cat((theta, x, t), dim=-1))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(-8.0 * t**2)

    def beta(self, t: Tensor) -> Tensor:
        return 16.0 * t

    def ode(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        drift = -self.beta(t) / 2 * theta
        diffusion = -self.beta(t) / 2 * (1 + self.alpha(t)) * self.forward(theta, x, t)

        return drift + diffusion

    def flow(self, x: Tensor) -> Distribution:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The posterior distribution :math:`p_\phi(\theta | x)` induced by the
            probability flow ODE.

        Note:
            The :func:`log_prob` method of the returned distribution is an unbiased
            stochastic estimator of the true log-density. One should average over
            a sufficient number of evaluations to attain small errors.
        """

        batch_shape = x.shape[:-1]

        return NormalizingFlow(
            transform=FFJTransform(
                f=lambda theta, t: self.ode(theta, x, t),
                time=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(batch_shape),
        )


class NSELoss(nn.Module):
    r"""Creates a module that calculates the rescaled score-matching loss for a NSE
    regressor.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \| s_\phi(\theta'_i, x_i, t_i) + \varepsilon_i \|_2^2

    where :math:`t_i \sim \mathcal{U}(0, 1)`, :math:`\varepsilon_i \sim \mathcal{N}(0,
    I)` and :math:`\theta'_i = \sqrt{\alpha(t_i)} \, \theta_i + (1 - \alpha(t_i)) \,
    \varepsilon_i`.

    Arguments:
        estimator: A regression network :math:`s_\phi(\theta, x, t)`.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        t = theta.new_empty(theta.shape[:-1]).uniform_(0, 1)
        alpha = self.estimator.alpha(t)[..., None]

        epsilon = torch.randn_like(theta)
        theta_prime = alpha.sqrt() * theta + (1 - alpha) * epsilon

        score = self.estimator(theta_prime, x, t)

        return (score + epsilon).square().mean()


class MetropolisHastings(object):
    r"""Creates a batched Metropolis-Hastings sampler.

    Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) sampling algorithm used to
    sample from intractable distributions :math:`p(x)` whose density is proportional to
    a tractable function :math:`f(x)`, with :math:`x \in \mathcal{X}`. The algorithm
    consists in repeating the following routine for :math:`t = 1` to :math:`T`, where
    :math:`x_0` is the initial sample and :math:`q(x' | x)` is a pre-defined transition
    distribution.

    .. math::
        1. ~ & x' \sim q(x' | x_{t-1}) \\
        2. ~ & \alpha \gets \frac{f(x')}{f(x_{t-1})}
            \frac{q(x_{t-1} | x')}{q(x' | x_{t-1})} \\
        3. ~ & u \sim \mathcal{U}(0, 1) \\
        4. ~ & x_t \gets \begin{cases}
            x' & \text{if } u \leq \alpha \\
            x_{t-1} & \text{otherwise}
        \end{cases}

    Asymptotically, i.e. when :math:`T \to \infty`, the distribution of samples
    :math:`x_t` is guaranteed to converge towards :math:`p(x)`. In this implementation,
    a Gaussian transition :math:`q(x' | x) = \mathcal{N}(x'; x, \Sigma)` is used, which
    can be modified by sub-classing :class:`MetropolisHastings`.

    Wikipedia:
        https://wikipedia.org/wiki/Metropolis-Hastings_algorithm

    Arguments:
        x_0: A batch of initial points :math:`x_0`, with shape :math:`(*, L)`.
        f: A function :math:`f(x)` proportional to a density function :math:`p(x)`.
        log_f: The logarithm :math:`\log f(x)` of a function proportional
            to :math:`p(x)`.
        sigma: The standard deviation of the Gaussian transition.
            Either a scalar or a vector.

    Example:
        >>> x_0 = torch.randn(128, 7)
        >>> log_f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> sampler = MetropolisHastings(x_0, log_f=log_f, sigma=0.5)
        >>> samples = [x for x in sampler(256, burn=128, step=4)]
        >>> samples = torch.stack(samples)
        >>> samples.shape
        torch.Size([32, 128, 7])
    """

    def __init__(
        self,
        x_0: Tensor,
        f: Callable[[Tensor], Tensor] = None,
        log_f: Callable[[Tensor], Tensor] = None,
        sigma: Union[float, Tensor] = 1.0,
    ):
        super().__init__()

        self.x_0 = x_0

        assert f is not None or log_f is not None, \
            "Either 'f' or 'log_f' has to be provided."

        if f is None:
            self.log_f = log_f
        else:
            self.log_f = lambda x: f(x).log()

        self.sigma = sigma

    def q(self, x: Tensor) -> Distribution:
        return DiagNormal(x, torch.ones_like(x) * self.sigma)

    @property
    def symmetric(self) -> bool:
        return True

    def __iter__(self) -> Iterator[Tensor]:
        x = self.x_0

        # log f(x)
        log_f_x = self.log_f(x)

        while True:
            # y ~ q(y | x)
            y = self.q(x).sample()

            # log f(y)
            log_f_y = self.log_f(y)

            #     f(y)   q(x | y)
            # a = ---- * --------
            #     f(x)   q(y | x)
            log_a = log_f_y - log_f_x

            if not self.symmetric:
                log_a = log_a + self.q(y).log_prob(x) - self.q(x).log_prob(y)

            a = log_a.exp()

            # u in [0; 1]
            u = torch.rand(a.shape).to(a)

            # if u < a, x <- y
            # else x <- x
            mask = u < a

            x = torch.where(mask.unsqueeze(-1), y, x)
            log_f_x = torch.where(mask, log_f_y, log_f_x)

            yield x

    def __call__(self, stop: int, burn: int = 0, step: int = 1) -> Iterable[Tensor]:
        return islice(self, burn, stop, step)
