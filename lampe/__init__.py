r"""Likelihood-free AMortized Posterior Estimation (LAMPE)"""

from . import patch
from .data import JointLoader, H5Dataset
from .nn import NRE, NPE
from .nn.losses import NRELoss, NPELoss
from .priors import BoxUniform, DiagNormal
