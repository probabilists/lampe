r"""Inference components such as estimators, training losses and MCMC samplers."""

__all__ = [
    'NRE',
    'NRELoss',
    'BNRELoss',
    'CNRELoss',
    'BCNRELoss',
    'AMNRE',
    'AMNRELoss',
    'NPE',
    'NPELoss',
    'FMPE',
    'FMPELoss',
    'MetropolisHastings',
]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import islice
from torch import Tensor, BoolTensor
from torch.distributions import Distribution
from typing import *

from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.flows import Flow, MAF
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from .nn import MLP


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) network.

    The principle of neural ratio estimation is to train a classifier network
    :math:`d_\phi(\theta, x)` to discriminate between pairs :math:`(\theta, x)` equally
    sampled from the joint distribution :math:`p(\theta, x)` and the product of the
    marginals :math:`p(\theta)p(x)`. Formally, the optimization problem is

    .. math:: \arg\min_\phi
        \frac{1}{2} \mathbb{E}_{p(\theta, x)} \big[ \ell(d_\phi(\theta, x)) \big] +
        \frac{1}{2} \mathbb{E}_{p(\theta)p(x)} \big[ \ell(1 - d_\phi(\theta, x)) \big]

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
        build: The network constructor (e.g. :class:`lampe.nn.ResMLP`). It takes the
            number of input and output features as positional arguments.
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
    r"""Creates a module that calculates the cross-entropy loss for a NRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
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

        return (l1 + l0) / 2


class BNRELoss(nn.Module):
    r"""Creates a module that calculates the balanced cross-entropy loss for a balanced
    NRE (BNRE) network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{2N} \sum_{i = 1}^N
            \ell(d_\phi(\theta_i, x_i)) + \ell(1 - d_\phi(\theta_{i+1}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\theta_i, x_i) + d_\phi(\theta_{i+1}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood.

    References:
        | Towards Reliable Simulation-Based Inference with Balanced Neural Ratio Estimation (Delaunoy et al., 2022)
        | https://arxiv.org/abs/2208.13624

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
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

        return (l1 + l0) / 2 + self.lmbda * lb


class CNRELoss(nn.Module):
    r"""Creates a module that calculates the cross-entropy loss for a contrastive NRE
    (CNRE) network.

    The principle of contrastive neural ratio estimation (CNRE) is to predict whether a
    set :math:`\Theta = \{\theta^1, \dots, \theta^K\}` contains or not the parameters
    that originated an observation :math:`x`. The elements of :math:`\Theta` are drawn
    independently from the prior :math:`p(\theta)` and the element :math:`\theta^k` that
    originates the observation :math:`x \sim p(x | \theta^k)` is chosen uniformly within
    :math:`\Theta`, such that

    .. math:: p(\Theta, x)
        & = p(\Theta) \, p(x | \Theta) \\
        & = p(\Theta) \frac{1}{K} \sum_{k = 1}^K p(x | \theta^k) \\
        & = p(\Theta) \, p(x) \frac{1}{K} \sum_{k = 1}^K r(\theta^k, x)

    where :math:`r(\theta, x)` is the likelihood-to-evidence (LTE) ratio. The task
    is to discriminate between pairs :math:`(\Theta, x)` for which :math:`\Theta` either
    does or does not contain the nominal parameters of :math:`x`, similar to the
    original NRE optimization problem. For this task, the decision function
    modeling the Bayes optimal classifier is

    .. math:: d(\Theta, x)
        = \frac{p(\Theta, x)}{p(\Theta, x) + \frac{1}{\gamma} p(\Theta) p(x)}
        = \frac{\sum_{k = 1}^K r(\theta^k, x)}{\frac{K}{\gamma} + \sum_{k = 1}^K r(\theta^k, x)} \, ,

    where :math:`\gamma \in \mathbb{R}^+` are the odds of :math:`\Theta` containing to
    not containing the nominal parameters. Consequently, a classifier
    :math:`d_\phi(\Theta, x)` can be equivalently replaced and trained as a composition
    of ratios :math:`r_\phi(\theta^k, x)`. Eventually, given a batch of :math:`N \geq
    2K` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \frac{\gamma}{\gamma + 1} \ell(d_\phi(\Theta_i, x_i))
        + \frac{1}{\gamma + 1} \ell(1 - d_\phi(\Theta_{i+K}, x_i))

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and :math:`\Theta_i =
    \{\theta_i, \dots, \theta_{i+K-1}\}`.

    Note:
        The quantity :math:`d_\phi(\Theta, x)` corresponds to :math:`q_\phi(y \neq 0 |
        \Theta, x)` or :math:`1 - q_\phi(y = 0 | \Theta, x)` in the notations of Miller
        et al. (2022).

    References:
        | Contrastive Neural Ratio Estimation (Miller et al., 2022)
        | https://arxiv.org/abs/2210.06170

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        cardinality: The cardinality :math:`K` of :math:`\Theta`.
        gamma: The odds ratio :math:`\gamma`.
    """

    def __init__(
        self,
        estimator: nn.Module,
        cardinality: int = 2,
        gamma: float = 1.0,
    ):
        super().__init__()

        self.estimator = estimator
        self.cardinality = cardinality
        self.gamma = gamma

    def logits(self, theta: Tensor, x: Tensor) -> Tensor:
        theta = torch.cat((theta, theta[: self.cardinality - 1]), dim=0)
        theta = theta.unfold(0, self.cardinality, 1)
        theta = theta.movedim(-1, 0)

        theta_prime = torch.roll(theta, self.cardinality, dims=1)

        log_r, log_r_prime = self.estimator(
            torch.stack((theta, theta_prime)),
            x,
        )

        shift = math.log(self.gamma / self.cardinality)

        log_r = torch.logsumexp(log_r, dim=0) + shift
        log_r_prime = torch.logsumexp(log_r_prime, dim=0) + shift

        return log_r, log_r_prime

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_r, log_r_prime = self.logits(theta, x)

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (self.gamma * l1 + l0) / (self.gamma + 1)


class BCNRELoss(CNRELoss):
    r"""Creates a module that calculates the balanced cross-entropy loss for a balanced
    CNRE (BCNRE) network.

    Given a batch of :math:`N \geq 2K` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math::
        l & = \frac{1}{N} \sum_{i = 1}^N
            \frac{\gamma}{\gamma + 1} \ell(d_\phi(\Theta_i, x_i))
            + \frac{1}{\gamma + 1} \ell(1 - d_\phi(\Theta_{i+K}, x_i)) \\
          & + \lambda \left(1 - \frac{1}{N} \sum_{i = 1}^N
            d_\phi(\Theta_i, x_i) + d_\phi(\Theta_{i+K}, x_i) \right)^2

    where :math:`\ell(p) = -\log p` is the negative log-likelihood and :math:`\Theta_i =
    \{\theta_i, \dots, \theta_{i+K-1}\}`.

    References:
        | Balancing Simulation-based Inference for Conservative Posteriors (Delaunoy et al., 2023)
        | https://arxiv.org/abs/2304.10978

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x)`.
        cardinality: The cardinality :math:`K` of :math:`\Theta`.
        gamma: The odds ratio :math:`\gamma`.
        lmbda: The weight :math:`\lambda` controlling the strength of the balancing
            condition.
    """

    def __init__(
        self,
        estimator: nn.Module,
        cardinality: int = 2,
        gamma: float = 1.0,
        lmbda: float = 100.0,
    ):
        super().__init__(estimator, cardinality, gamma)

        self.lmbda = lmbda

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`l`.
        """

        log_r, log_r_prime = self.logits(theta, x)

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()
        lb = (torch.sigmoid(log_r) + torch.sigmoid(log_r_prime) - 1).mean().square()

        return (self.gamma * l1 + l0) / (self.gamma + 1) + self.lmbda * lb


class AMNRE(NRE):
    r"""Creates an arbitrary marginal neural ratio estimation (AMNRE) network.

    The principle of AMNRE is to introduce, as input to the classifier, a binary mask
    :math:`b \in \{0, 1\}^D` indicating a subset of parameters :math:`\theta_b =
    (\theta_i: b_i = 1)` of interest. Intuitively, this allows the classifier to
    distinguish subspaces and to learn a different ratio for each of them. Formally, the
    classifier network takes the form :math:`d_\phi(\theta_b, x, b)` and the
    optimization problem becomes

    .. math:: \arg\min_\phi
        \frac{1}{2} \mathbb{E}_{p(\theta, x) P(b)} \big[ \ell(d_\phi(\theta_b, x, b)) \big] +
        \frac{1}{2} \mathbb{E}_{p(\theta)p(x) P(b)} \big[ \ell(1 - d_\phi(\theta_b, x, b)) \big],

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
    r"""Creates a module that calculates the cross-entropy loss for an AMNRE network.

    Given a batch of :math:`N \geq 2` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
        \ell(d_\phi(\theta_i \odot b_i, x_i, b_i)) +
        \ell(1 - d_\phi(\theta_{i+1} \odot b_i, x_i, b_i))

    where the binary masks :math:`b_i` are sampled from a distribution :math:`P(b)`.

    Arguments:
        estimator: A log-ratio network :math:`\log r_\phi(\theta, x, b)`.
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

        return (l1 + l0) / 2


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
        build: The flow constructor (e.g. :class:`zuko.flows.spline.NSF`). It takes
            the number of sample and context features as positional arguments.
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], Flow] = MAF,
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


class FMPE(nn.Module):
    r"""Creates a flow matching posterior estimation (FMPE) network.

    The principle of FMPE is to train a regression network :math:`v_\phi(\theta, x, t)`
    to approximate a vector field inducing a time-continuous normalizing flow between
    the posterior distribution :math:`p(\theta | x)` and a standard Gaussian
    distribution :math:`\mathcal{N}(0, I)`.

    After training, the normalizing flow :math:`p_\phi(\theta | x)` induced by
    :math:`v_\phi(\theta, x, t)` is used to evaluate the posterior density or generate
    samples.

    References:
        | Flow Matching for Generative Modeling (Lipman et al., 2023)
        | https://arxiv.org/abs/2210.02747

        | Flow Matching for Scalable Simulation-Based Inference (Dax et al., 2023)
        | https://arxiv.org/abs/2305.17161

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

        self.register_buffer('freqs', torch.arange(1, freqs + 1) * math.pi)
        self.register_buffer('zeros', torch.zeros(theta_dim))
        self.register_buffer('ones', torch.ones(theta_dim))

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

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N\|
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

        assert (
            f is not None or log_f is not None
        ), "Either 'f' or 'log_f' has to be provided."

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
