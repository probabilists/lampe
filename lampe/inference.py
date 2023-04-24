r"""Inference components such as estimators, training losses and MCMC samplers."""

__all__ = [
    'NRE',
    'NRELoss',
    'BNRELoss',
    'CNRELoss',
    'BinaryBalancedCNRELoss',
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
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from .nn import MLP


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) classifier network.

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
    r"""Creates a module that calculates the cross-entropy loss for a NRE classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
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

        return (l1 + l0) / 2


class BNRELoss(nn.Module):
    r"""Creates a module that calculates the balanced cross-entropy loss for a balanced
    NRE (BNRE) classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

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

        return (l1 + l0) / 2 + self.lmbda * lb


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
    r"""Creates a module that calculates the cross-entropy loss for an AMNRE classifier.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{2N} \sum_{i = 1}^N
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
        build: The flow constructor (e.g. :class:`zuko.flows.NSF`). It takes the
            number of sample and context features as positional arguments.
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


class CNRELoss(nn.Module):
    r"""Creates a module that calculates the loss :math:`l_{\gamma, K}` of a 
    NRE classifier :math:`q_{w}(y \mid \theta, x)` from Contrastive Neural 
    Ratio Estimation [1]. 
    
    This is a cross-entropy loss (via ''multi-class sigmoid'' activation) for
    1-out-of-`K + 1` classification. It requires samples:
    :math:`\Theta^{(i)} \coloneqq (\theta_{1}^{(i)}, \ldots, \theta_{K}^{(i)}) \sim p(\theta)`,
    :math:`x^{(i)} \sim p(x \mid \theta_k)`,
    :math:`\Theta^{(i')} \coloneqq (\theta_{1}^{(i')}, \ldots, \theta_{K}^{(i')}) \sim p(\theta)`,
    and :math:`x^{(i')} \sim p(x)`
    with :math:`i = 1, 2, \ldots, N` and :math:`i' = 1, 2, \ldots, N`.
    
    In practice, the independent samples are drawn from applying `torch.roll` 
    :math:`2K - 1` times to a batch of :math:`N` pairs 
    :math:`(\theta^{(n)}, x^{(n)}) \sim p(\theta, x)`.
    
    Let :math:`\Theta \coloneqq (\theta_1, ..., \theta_K)`, we define the 
    classifier:

    .. math::
        q_{w}(y = k \mid \Theta, \bx) \coloneqq
        \begin{cases}
            \frac{K}{K + \gamma \sum_{j=1}^{K} \exp \circ h_{w}(\theta_j, \bx)} & k = 0 \\
            \frac{\gamma \, \exp \circ h_{w}(\theta_k,\bx))}{K + \gamma \sum_{j=1}^{K} \exp \circ h_{w}(\theta_j,\bx)} & k = 1, \ldots, K.
        \end{cases}
    
    where :math:`h_{w}` is a neural network with weights :math:`w` and 
    :math:`\gamma \coloneqq \frac{K p(y=k)}{p(y=1)}` defines the target
    distribution over :math:`y`.
    
    The module returns

    .. math::
        \hat{\ell}_{\gamma, K}(w) & \coloneqq 
            -\frac{1}{N} \Bigg[
                \frac{1}{1 + \gamma} \frac{1}{N} \sum_{i = 1}^{N} q_{w}(y = 0 \mid \Theta^{(i)}, \bx^{(i)})
                \frac{\gamma}{1 + \gamma} \frac{1}{N} \sum_{i' = 1}^{N} q_{w}(y = K \mid \Theta^{(i')}, \bx^{(i')})
            \Bigg].

    References:
        [1] Benajmin Kurt Miller, et. al., _Contrastive Neural Ratio Estimation_,
            NeurIPS 2022, https://arxiv.org/abs/2210.06170
    
    Arguments:
        estimator: The log ratio estimator :math:`h_w(\theta, x)`.
        num_classes: The number of classes :math:`K`.
        gamma: The ratio of the distribution over possible classes, specifically
            :math:`\gamma \coloneqq \frac{K p(y=k)}{p(y=0)}`.
    """

    def __init__(self, estimator: nn.Module, num_classes: int, gamma: float):
        super().__init__()
        self.estimator = estimator
        assert num_classes >= 1, f"num_classes = {num_classes} must be greater than 1."
        self.num_classes = num_classes
        assert gamma > 0, f"gamma = {gamma} must be greater than 0."
        self.gamma = gamma
    
    def _get_log_rs(
        self,
        theta: Tensor, 
        x: Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            (log_r_y_K, log_r_y_0): log_r_y_K the log ratios computed for y=K 
                with shape :math:`(K, N)`, log_r_y_0 the log ratios computed 
                for y=0 with shape :math:`(K, N)`.
        """
        assert theta.shape[0] == x.shape[0], "Batch sizes for theta and x must match."
        batch_size = theta.shape[0]

        inds = torch.arange(batch_size)
        two_k_rolled_copies = torch.arange(2 * self.num_classes)
        rolled_inds = (inds + two_k_rolled_copies[:, None]) % batch_size
        log_r = self.estimator(theta[rolled_inds], x)
        
        # index [0, ...] of log_r_y_K is the theta-x-pair sampled from the joint p(theta,x)
        # all other indicies are drawn marginally
        # log_r_y_0 only contains marginal draws
        log_r_y_K, log_r_y_0 = log_r.split((self.num_classes, self.num_classes))
        return log_r_y_K, log_r_y_0

    def _get_log_probs(
        self,
        log_r_y_K: torch.Tensor, 
        log_r_y_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Arguments:
            log_r_y_K: the log ratios computed for y=K with shape :math:`(K, N)`.
            log_r_y_0: the log ratios computed for y=0 with shape :math:`(K, N)`.

        Returns:
            (log_prob_y_K, log_prob_y_0): log_prob_y_K is the classification log prob for y=K,
                log_prob_y_0 is the classification log prob for y=0
        """
        batch_size = log_r_y_K.shape[-1]

        dtype = log_r_y_K.dtype
        device = log_r_y_K.device

        # To use logsumexp, we extend the denominator logits with loggamma
        loggamma = torch.tensor(self.gamma, dtype=dtype, device=device).log()
        logK = torch.tensor(self.num_classes, dtype=dtype, device=device).log()
        denominator_y_K = torch.concat(
            [loggamma + log_r_y_K, logK.expand((1, batch_size))],
            dim=0,
        )
        denominator_y_0 = torch.concat(
            [loggamma + log_r_y_0, logK.expand((1, batch_size))],
            dim=0,
        )

        # Compute the contributions to the loss from each term in the classification.
        log_prob_y_K = (
            loggamma + log_r_y_K[0, :] - torch.logsumexp(denominator_y_K, dim=0)
        )
        log_prob_y_0 = logK - torch.logsumexp(denominator_y_0, dim=0)

        return log_prob_y_K, log_prob_y_0

    def forward(
        self, theta: Tensor, x: Tensor
    ) -> torch.Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`\hat{\ell}_{\gamma, K}(w)`.
        """
        log_r_y_K, log_r_y_0 = self._get_log_rs(theta, x)
        log_prob_y_K, log_prob_y_0 = self._get_log_probs(log_r_y_K, log_r_y_0)

        return -torch.mean(
            (1 / (1 + self.gamma)) * log_prob_y_0 + \
            (self.gamma / (1 + self.gamma)) * log_prob_y_K
        )


class BinaryBalancedCNRELoss(CNRELoss):
    r"""Creates a module that calculates the loss :math:`l_{\gamma, K, \lambda}` 
    of a NRE classifier :math:`q_{w}(y \mid \theta, x)` from Contrastive Neural 
    Ratio Estimation [1], but regularized by the balance criterion from 
    Balancing Simulation-based Inference for Conservative Posteriors [1]. 
    Further details in Appendix B of [1].

    Given a batch of :math:`N` pairs :math:`(\theta^{(n)}, x^{(n)})`, the module 
    returns
    :math:`\hat{\ell}_{\gamma, K}(w) + \lambda \hat{\ell}_{B}(w)`. 
    The balance criterion is defined

    .. math::
        \hat{\ell}_{B}(w) \coloneqq
            \left(1 - \frac{1}{N} \sum_{i = n}^{N}
            \sigma \circ h_{w}(\theta^{(n)}, x^{(n)}) + 
            \sigma \circ h_{w}(\theta^{(n')}, x^{(n')}) \right)^2

    where :math:`\sigma` is the logistic sigmoid, :math:`h_{w}` is the neural 
    network, :math:`\theta^{(n)}, x^{(n)} \sim p(\theta, x)` and 
    :math:`\theta^{(n')}, x^{(n')} \sim p(\theta)p(x)`. In practice, the 
    independent samples are drawn from applying `torch.roll` to the batch.
    
    References:
        [1] Arnaud Delaunoy, Benjamin Kurt Miller, et. al., 
            _Balancing Simulation-based Inference for Conservative Posteriors_,
            https://arxiv.org/abs/2304.10978

    Arguments:
        estimator: The log ratio estimator :math:`h_w(\theta, x)`.
        num_classes: The number of classes :math:`K`.
        gamma: The ratio of the distribution over possible classes, specifically
            :math:`\gamma \coloneqq \frac{K p(y=k)}{p(y=0)}`.
        lmbda: lagrange multiplier for balance condition
    """

    def __init__(self, estimator: nn.Module, num_classes: int, gamma: float, lmbda: float):
        super().__init__(estimator, num_classes, gamma)
        self.lmbda = lmbda

    def forward(
        self, theta: Tensor, x: Tensor
    ) -> torch.Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(N, D)`.
            x: The observation :math:`x`, with shape :math:`(N, L)`.

        Returns:
            The scalar loss :math:`\hat{\ell}_{\gamma, K}(w) + \lambda \hat(\ell)_{B}(w)`.
        """
        log_r_y_K, log_r_y_0 = self._get_log_rs(theta, x)
        log_prob_y_K, log_prob_y_0 = self._get_log_probs(log_r_y_K, log_r_y_0)

        return -torch.mean(
            (1 / (1 + self.gamma)) * log_prob_y_0 + \
            (self.gamma / (1 + self.gamma)) * log_prob_y_K
        ) + self.lmbda * \
        (
            torch.sigmoid(log_r_y_K[0, :]) + \
            torch.sigmoid(log_r_y_0[0, :]) - 1
        ).mean().square()


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
            The rescaled score :math:`s_\phi(\theta, x, t)`, with shape :math:`(*, D)`.
        """

        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        theta, x, t = broadcast(theta, x, t, ignore=1)

        return self.net(torch.cat((theta, x, t), dim=-1))

    def alpha(self, t: Tensor) -> Tensor:
        return torch.exp(-8.0 * t**2)

    def beta(self, t: Tensor) -> Tensor:
        return 16.0 * t

    def ode(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        alpha, beta = self.alpha(t), self.beta(t)
        score = self.forward(theta, x, t)

        return -beta / 2 * (theta + (1 + alpha) * score)

    def flow(self, x: Tensor) -> Distribution:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The posterior distribution :math:`p_\phi(\theta | x)` induced by the
            probability flow ODE.
        """

        return NormalizingFlow(
            transform=FreeFormJacobianTransform(
                f=lambda t, theta: self.ode(theta, x, t),
                time=x.new_tensor(1.0),
                phi=(x, *self.parameters()),
            ),
            base=DiagNormal(self.zeros, self.ones).expand(x.shape[:-1]),
        )


class NSELoss(nn.Module):
    r"""Creates a module that calculates the rescaled score-matching loss for a NSE
    regressor.

    Given a batch of :math:`N` pairs :math:`(\theta_i, x_i)`, the module returns

    .. math:: l = \frac{1}{N} \sum_{i = 1}^N
        \| s_\phi(\theta'_i, x_i, t_i) + z_i \|_2^2

    where :math:`t_i \sim \mathcal{U}(0, 1)`, :math:`z_i \sim \mathcal{N}(0,
    I)` and :math:`\theta'_i = \sqrt{\alpha(t_i)} \, \theta_i + (1 - \alpha(t_i)) \,
    z_i`.

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

        z = torch.randn_like(theta)
        theta_prime = alpha.sqrt() * theta + (1 - alpha) * z

        score = self.estimator(theta_prime, x, t)

        return (score + z).square().mean()


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
