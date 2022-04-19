r"""Neural networks, layers and modules.

.. admonition:: TODO

    * Finish documentation (NPE, AMNPE).
    * Find references.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from torch.distributions import Distribution
from typing import *

from .flows import MAF
from ..utils import broadcast


__all__ = [
    'MLP', 'ResBlock', 'ResMLP',
    'NRE', 'AMNRE', 'NPE', 'AMNPE',
]


class Affine(nn.Module):
    r"""Creates an element-wise affine layer.

    Arguments:
        shift: The shift term.
        scale: The scale factor.
    """

    def __init__(self, shift: Tensor, scale: Tensor):
        super().__init__()

        self.register_buffer('shift', shift)
        self.register_buffer('scale', scale)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale + self.shift

    def extra_repr(self) -> str:
        return '\n'.join([
            f'(shift): {self.shift.cpu()}',
            f'(scale): {self.scale.cpu()}',
        ])


class BatchNorm0d(nn.BatchNorm1d):
    r"""Creates a batch normalization (BatchNorm) layer for scalars.

    References:
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        (Ioffe et al., 2015)
        https://arxiv.org/abs/1502.03167

    Arguments:
        args: Positional arguments passed to :class:`torch.nn.BatchNorm1d`.
        kwargs: Keyword arguments passed to :class:`torch.nn.BatchNorm1d`.
    """

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape

        x = x.reshape(-1, shape[-1])
        x = super().forward(x)
        x = x.reshape(shape)

        return x


class MLP(nn.Sequential):
    r"""Creates a multi-layer perceptron (MLP).

    Also known as fully connected feedforward network, an MLP is a sequence of
    non-linear parametric transformations

    .. math:: h_{i + 1} = a_{i + 1}(W_{i + 1}^T h_i + b_{i + 1}),

    over feature vectors :math:`h_i`, with the input and ouput feature vectors
    :math:`x = h_0` and :math:`y = h_L`, respectively. The non-linear functions
    :math:`a_i` are called activation functions. The trainable parameters of an MLP
    are its weights and biases :math:`\phi = \{W_i, b_i | i = 1, \dots, L\}`.

    Wikipedia:
        https://en.wikipedia.org/wiki/Feedforward_neural_network

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        activation: The activation layer type.
        batchnorm: Whether to use batch normalization or not.
        dropout: The dropout rate.
        kwargs: Keyword arguments passed to :class:`torch.nn.Linear`.

    Example:
        >>> net = MLP(64, 1, [32, 16], activation='ELU')
        >>> net
        MLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        activation: str = 'ReLU',
        batchnorm: bool = False,
        dropout: float = 0.,
        **kwargs,
    ):
        activation = {
            'ReLU': nn.ReLU,
            'ELU': nn.ELU,
            'CELU': nn.CELU,
            'SELU': nn.SELU,
            'GELU': nn.GELU,
        }.get(activation, nn.ReLU)

        batchnorm = BatchNorm0d if batchnorm else lambda _: None
        dropout = nn.Dropout(dropout) if dropout > 0 else None

        layers = []

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([
                nn.Linear(before, after, **kwargs),
                batchnorm(after),
                activation(),
                dropout,
            ])

        layers = layers[:-3]
        layers = filter(lambda l: l is not None, layers)

        super().__init__(*layers)

        self.in_features = in_features
        self.out_features = out_features


class ResBlock(MLP):
    r"""Creates a residual block.

    A residual block is a function of the type

    .. math:: y = x + f(x),

    where :math:`f` is a non-linear parametric transformation. An MLP with a
    constant number of features in hidden layers is commonly used as :math:`f`.

    Arguments:
        features: The input, output and hidden features.
        hidden_layers: The number of hidden layers.
        kwargs: Keyword arguments passed to :class:`MLP`.

    Example:
        >>> net = ResBlock(32, hidden_layers=3, activation='ELU')
        >>> net
        ResBlock(
          (0): Linear(in_features=32, out_features=32, bias=True)
          (1): ELU(alpha=1.0)
          (2): Linear(in_features=32, out_features=32, bias=True)
          (3): ELU(alpha=1.0)
          (4): Linear(in_features=32, out_features=32, bias=True)
          (5): ELU(alpha=1.0)
          (6): Linear(in_features=32, out_features=32, bias=True)
        )
    """

    def __init__(self, features: int, hidden_layers: int = 2, **kwargs):
        super().__init__(
            features,
            features,
            [features] * hidden_layers,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResMLP(nn.Sequential):
    r"""Creates a residual multi-layer perceptron (ResMLP).

    Like the regular MLP, the ResMLP is a sequence of non-linear parametric
    transformations. However, it uses residual blocks as transformations, which
    reduces the vanishing of gradients and allows for deeper networks.

    Arguments:
        in_features: The number of input features.
        out_features: The number of output features.
        hidden_features: The numbers of hidden features.
        kwargs: Keyword arguments passed to :class:`ResBlock`.

    Example:
        >>> net = ResMLP(64, 1, [32, 16], activation='ELU')
        >>> net
        ResMLP(
          (0): Linear(in_features=64, out_features=32, bias=True)
          (1): ResBlock(
            (0): Linear(in_features=32, out_features=32, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=32, out_features=32, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=32, out_features=32, bias=True)
          )
          (2): Linear(in_features=32, out_features=16, bias=True)
          (3): ResBlock(
            (0): Linear(in_features=16, out_features=16, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=16, out_features=16, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=16, out_features=16, bias=True)
          )
          (4): Linear(in_features=16, out_features=1, bias=True)
        )
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
        **kwargs,
    ):
        blocks = []

        for before, after in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            if after != before:
                blocks.append(nn.Linear(before, after))

            blocks.append(ResBlock(after, **kwargs))

        blocks = blocks[:-1]

        super().__init__(*blocks)

        self.in_features = in_features
        self.out_features = out_features


class NRE(nn.Module):
    r"""Creates a neural ratio estimation (NRE) classifier network.

    The principle of neural ratio estimation is to train a classifier network
    :math:`d_\phi(\theta, x)` to discriminate between pairs :math:`(\theta, x)`
    equally sampled from the joint distribution :math:`p(\theta, x)` and the
    product of the marginals :math:`p(\theta)p(x)`. Formally, the optimization
    problem is

    .. math:: \arg \min_\phi
        \mathbb{E}_{p(\theta, x)} \big[ \ell(d_\phi(\theta, x)) \big] +
        \mathbb{E}_{p(\theta)p(x)} \big[ \ell(1 - d_\phi(\theta, x)) \big]

    where :math:`\ell(p) = - \log p` is the negative log-likelihood.
    For this task, the decision function modeling the Bayes optimal classifier is

    .. math:: d(\theta, x)
        = \frac{p(\theta, x)}{p(\theta, x) + p(\theta) p(x)}

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
        Approximating Likelihood Ratios with Calibrated Discriminative Classifiers
        (Cranmer et al., 2015)
        https://arxiv.org/abs/1506.02169

        Likelihood-free MCMC with Amortized Approximate Ratio Estimators
        (Hermans et al., 2019)
        https://arxiv.org/abs/1903.04057

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        moments: The parameters moments :math:`\mu` and :math:`\sigma` for standardization.
        const: The network constructor (e.g. :class:`MLP` or :class:`ResMLP`).
        kwargs: Keyword arguments passed to the constructor.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        moments: Tuple[Tensor, Tensor] = None,
        const: Callable[[int, int], nn.Module] = MLP,
        **kwargs,
    ):
        super().__init__()

        if moments is not None:
            mu, sigma = moments

        self.standardize = nn.Identity() if moments is None else Affine(-mu / sigma, 1 / sigma)

        self.net = const(theta_dim + x_dim, 1, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta, x)`, with shape :math:`(*,)`.
        """

        theta = self.standardize(theta)
        theta, x = broadcast(theta, x, ignore=1)

        return self.net(torch.cat((theta, x), dim=-1)).squeeze(-1)


class AMNRE(NRE):
    r"""Creates an arbitrary marginal neural ratio estimation (AMNRE) classifier
    network.

    The principle of AMNRE is to introduce, as input to the classifier, a binary mask
    :math:`b \in \{0, 1\}^D` indicating a subset of parameters :math:`\theta_b =
    (\theta_i: b_i = 1)` of interest. Intuitively, this allows the classifier to
    distinguish subspaces and to learn a different ratio for each of them. Formally,
    the classifer network takes the form :math:`d_\phi(\theta_b, x, b)` and the
    optimization problem becomes

    .. math:: \arg \min_\phi
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
        Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference
        (Rozet et al., 2021)
        https://arxiv.org/abs/2110.00449

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        args: Positional arguments passed to :class:`NRE`.
        kwargs: Keyword arguments passed to :class:`NRE`.
    """

    def __init__(
        self,
        theta_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(theta_dim * 2, *args, **kwargs)

    def forward(self, theta: Tensor, x: Tensor, b: BoolTensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`, or
                a subset :math:`\theta_b`, with shape :math:`(*, |b|)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(*, D)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta_b, x, b)`, with shape :math:`(*,)`.
        """

        zeros = theta.new_zeros(theta.shape[:-1] + b.shape[-1:])

        if b.dim() == 1 and theta.shape[-1] < b.numel():
            theta = zeros.masked_scatter(b, theta)
        else:
            theta = torch.where(b, theta, zeros)

        theta = self.standardize(theta) * b
        theta, x, b = broadcast(theta, x, b * 2. - 1., ignore=1)

        return self.net(torch.cat((theta, x, b), dim=-1)).squeeze(-1)


class NPE(nn.Module):
    r"""Creates a neural posterior estimation (NPE) normalizing flow.

    TODO

    Arguments:
        theta_dim: The dimensionality :math:`D` of the parameter space.
        x_dim: The dimensionality :math:`L` of the observation space.
        moments: The parameters moments :math:`\mu` and :math:`\sigma` for standardization.
        kwargs: Keyword arguments passed to :class:`flows.MAF`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        moments: Tuple[Tensor, Tensor] = None,
        **kwargs,
    ):
        super().__init__()

        self.flow = MAF(theta_dim, x_dim, moments=moments, **kwargs)

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-density :math:`\log p_\phi(\theta | x)`, with shape :math:`(*,)`.
        """

        theta, x = broadcast(theta, x, ignore=1)

        return self.flow.log_prob(theta, x)

    def sample(self, x: Tensor, shape: torch.Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            shape: TODO

        Returns:
            The samples :math:`\theta \sim p_\phi(\theta | x)`,
            with shape :math:`(*, S, D)`.
        """

        return self.flow.sample(x, shape)


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

        theta, x, b = broadcast(theta, x, b * 2. - 1., ignore=1)

        return self.flow.log_prob(theta, torch.cat((x, b), dim=-1))

    def sample(self, x: Tensor, b: BoolTensor, shape: torch.Size = ()) -> Tensor:
        r"""
        Arguments:
            x: The observation :math:`x`, with shape :math:`(*, L)`.
            b: A binary mask :math:`b`, with shape :math:`(D,)`.
            shape: TODO

        Returns:
            The samples :math:`\theta_b \sim p_\phi(\theta_b | x, b)`,
            with shape :math:`(*, S, D)`.
        """

        x, b = broadcast(x, b * 2. - 1., ignore=1)

        return self.flow.sample(torch.cat((x, b), dim=-1), shape)[..., b]
