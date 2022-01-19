r"""Losses and criteria"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, BoolTensor
from typing import *


def reduce(x: Tensor, reduction: str) -> Tensor:
    if reduction == 'sum':
        x = x.sum()
    elif reduction == 'mean':
        x = x.mean()
    elif reduction == 'batchmean':
        x = x.sum() / x.size(0)

    return x


class MSELoss(nn.Module):
    r"""Mean Squared Error (MSE) loss"""

    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        weight: Tensor = None,
    ) -> Tensor:
        error = F.mse_loss(input, target.detach(), reduction='none')

        if weight is not None:
            error = error * weight

        return reduce(error, self.reduction)


class RRLoss(MSELoss):
    r"""Ratio Regression (RR) loss

    (r - r*)^2
    """

    def forward(
        self,
        ratio: Tensor,  # log r
        target: Tensor,  # log r*
        weight: Tensor = None,
    ) -> Tensor:
        ratio, target = ratio.exp(), target.exp()

        return super().forward(ratio, target, weight)


class SRLoss(MSELoss):
    r"""Score Regression (SR) loss

    ||grad log r - grad log r*||^2
    """

    @staticmethod
    def score(
        theta: Tensor,  # theta
        ratio: Tensor,  # log r
    ) -> Tensor:
        return torch.autograd.grad(  # grad log r
            ratio, theta,
            torch.ones_like(ratio),
            create_graph=True,
        )[0]

    def forward(
        self,
        theta: Tensor,  # theta
        ratio: Tensor,  # log r
        target: torch.Tensor,  # log r*
        weight: Tensor = None,
    ) -> torch.Tensor:
        score = self.score(theta, ratio)
        target = self.score(theta, target)

        return super().forward(score, target, weight)


class NLLLoss(nn.Module):
    r"""Negative Log-Likelihood (NLL) loss

    - log x
    """

    def __init__(self, reduction: str = 'batchmean'):
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        log_prob: Tensor,  # log p
        weight: Tensor = None,
    ) -> Tensor:
        nll = -log_prob

        if weight is not None:
            nll = nll * weight

        return reduce(nll, self.reduction)


class NLLWithLogitsLoss(nn.Module):
    r"""Negative Log-Likelihood (NLL) with logits

    - log d(x)
    """

    def forward(self, logit: Tensor) -> Tensor:
        ld = F.logsigmoid(logit)  # log d(x)
        return -ld


class FocalWithLogitsLoss(nn.Module):
    r"""Focal Loss (FL) with logits

    - (1 - d(x))^gamma log d(x)

    References:
        [1] Focal Loss for Dense Object Detection
        (Lin et al., 2017)
        https://arxiv.org/abs/1708.02002

        [2] Calibrating Deep Neural Networks using Focal Loss
        (Mukhoti et al., 2020)
        https://arxiv.org/abs/2002.09437
    """

    def __init__(self, gamma: float = 2.):
        super().__init__()

        self.gamma = gamma

    def forward(self, logit: Tensor) -> Tensor:
        ld = F.logsigmoid(logit)  # log d(x)
        return -(1 - ld.exp()) ** self.gamma * ld


class PeripheralWithLogitsLoss(FocalWithLogitsLoss):
    r"""Peripheral Loss (PL) with logits

    - (1 - d(x)^gamma) log d(x)

    References:
        [1] Arbitrary Marginal Neural Ratio Estimation for Likelihood-free Inference
        (Rozet et al., 2021)
        https://matheo.uliege.be/handle/2268.2/12993
    """

    def forward(self, logit: Tensor) -> Tensor:
        ld = F.logsigmoid(logit)  # log d(x)
        return -(1 - (ld * self.gamma).exp()) * ld


class QSWithLogitsLoss(nn.Module):
    r"""Quadratic Score (QS) with logits

    (1 - d(x))^2

    References:
        https://en.wikipedia.org/wiki/Scoring_rule
    """

    def forward(self, logit: Tensor, weight: Tensor = None) -> Tensor:
        d = F.sigmoid(logit)  # d(x)
        return (1 - d) ** 2


SCORES = {
    'NLL': NLLWithLogitsLoss,
    'FL': FocalWithLogitsLoss,
    'PL': PeripheralWithLogitsLoss,
    'QS': QSWithLogitsLoss,
}


class BCEWithLogitsLoss(nn.Module):
    r"""Binary Cross-Entropy (BCE) loss with logits

    E_p [-log d(x)] + E_q [-log (1 - d(x))]

    Supports several scoring rules (NLL, PL, QS, ...).

    Wikipedia:
        https://en.wikipedia.org/wiki/Scoring_rule
    """

    def __init__(
        self,
        positive: str = 'NLL',  # in ['NLL', 'FL', 'PL', 'QS']
        negative: str = 'NLL',  # in ['NLL', 'FL', 'PL', 'QS']
        reduction: str = 'batchmean',
    ):
        super().__init__()

        self.l1 = SCORES[positive]()
        self.l0 = SCORES[negative]()

        self.reduction = reduction

    def forward(
        self,
        logit: Tensor,
        target: Tensor,
        weight: Tensor = None,
    ) -> Tensor:
        pos = target > 0.5

        l1 = self.l1(logit[pos])  # -log d(x)
        l0 = self.l0(-logit[~pos])  # -log (1 - d(x))

        if weight is not None:
            l1 = l1 * weight[pos]
            l0 = l0 * weight[~pos]

        cross = torch.cat((l1, l0))

        return reduce(cross, self.reduction)


class BalancingWithLogitsLoss(nn.Module):
    r"""Balancing loss

    (E_p [d(x)] + E_q [d(x)] - 1) ** 2
    """

    def forward(
        self,
        logit: Tensor,
        weight: Tensor = None,
    ) -> Tensor:
        d = torch.sigmoid(logit)  # d(x)

        if weight is None:
            d = d.mean()
        else:
            d = (weight * d).sum() / weight.sum()

        return (2 * d - 1) ** 2
