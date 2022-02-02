r"""Exoplanet Emission Spectrum (EES)

EES computes an emission spectrum based on disequilibrium carbon chemistry,
equilibrium clouds and a spline temperature-pressure profile of the exoplanet atmosphere.

References:
    [1] Retrieving scattering clouds and disequilibrium chemistry in the atmosphere of HR 8799e
    (Mollière et al., 2020)
    https://arxiv.org/abs/2006.09394

Shapes:
    theta: (16,)
    x: (947,)
"""

import numpy as np
import os
import torch
import warnings

try:
    os.environ['pRT_input_data_path'] = os.path.expanduser('~/.cache/lampe/pRT/data')

    import petitRADTRANS as prt
    import petitRADTRANS.retrieval.models as models
    import petitRADTRANS.retrieval.parameter as prm
except Exception as e:
    print(f'ImportWarning: {e}. \'EES\' requires')
    print('  conda install -c conda-forge multinest gfortran')
    print('  pip install petitRADTRANS')

from numpy import ndarray as Array
from torch import Tensor, BoolTensor
from typing import *

from . import Simulator
from ..priors import Distribution, JointUniform
from ..utils import cache, vectorize


labels = [
    f'${l}$' for l in [
        r'{\rm C/O}', r'\left[{\rm Fe/H}\right]', r'\log P_{\rm quench}',
        r'\log X_{\rm Fe}', r'\log X_{\rm MgSiO_3}',
        r'f_{\rm sed}', r'\log K_{zz}', r'\sigma_g', r'\log g', r'R_P',
        r'T_0', r'\frac{T_3}{T_{\rm connect}}', r'\frac{T_2}{T_3}', r'\frac{T_1}{T_2}',
        r'\alpha', r'\frac{\log \delta}{\alpha}',
    ]
]


bounds = torch.tensor([
    [0.1, 1.6],     # C/O
    [-1.5, 1.5],    # [Fe/H]
    [-6., 3.],      # log P_quench
    [-2.3, 1.],     # log X_Fe
    [-2.3, 1.],     # log X_MgSiO3
    [0., 10.],      # f_sed
    [5., 13.],      # log K_zz
    [1.05, 3.],     # sigma_g
    [2., 5.5],      # log g
    [0.9, 2.],      # R_P / R_Jupyter
    [300., 2300.],  # T_0 [K]
    [0., 1.],       # ∝ T_3 / T_connect
    [0., 1.],       # ∝ T_2 / T_3
    [0., 1.],       # ∝ T_1 / T_2
    [1., 2.],       # alpha
    [0., 1.],       # ∝ log delta / alpha
])

lower, upper = bounds[:, 0], bounds[:, 1]


def ees_prior(mask: BoolTensor = None) -> Distribution:
    r""" p(theta) """

    if mask is None:
        mask = ...

    return JointUniform(lower[mask], upper[mask])


class EES(Simulator):
    r"""Exoplanet Emission Spectrum (EES) simulator"""

    def __init__(self, log_scale: bool = True, **kwargs):
        super().__init__()

        self.log_scale = log_scale

        default = {
            'D_pl': 41.2925 * prt.nat_cst.pc,
            'pressure_scaling': 10,
            'pressure_simple': 100,
            'pressure_width': 3,
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        self.atmosphere = cache(prt.Radtrans, disk=True)(
            line_species=[
                'H2O_HITEMP',
                'CO_all_iso_HITEMP',
                'CH4',
                'NH3',
                'CO2',
                'H2S',
                'VO',
                'TiO_all_Exomol',
                # 'FeH',
                'PH3',
                'Na_allard',
                'K_allard',
            ],
            cloud_species=['MgSiO3(c)_cd', 'Fe(c)_cd'],
            rayleigh_species=['H2', 'He'],
            continuum_opacities=['H2-H2', 'H2-He'],
            wlen_bords_micron=[0.95, 2.45],
            do_scat_emis=True,
        )

        levels = (
            self.constants['pressure_simple'] + len(self.atmosphere.cloud_species) *
            (self.constants['pressure_scaling'] - 1) * self.constants['pressure_width']
        )

        self.atmosphere.setup_opa_structure(np.logspace(-6, 3, levels))

    def __call__(self, theta: Array) -> Array:
        r""" x ~ p(x | theta) """

        theta = {
            key: theta[..., i]
            for i, key in enumerate([
                'CO', 'FeH', 'log_pquench', 'log_X_Fe', 'log_X_MgSiO3', 'fsed', 'log_kzz',
                'sigma_lnorm', 'log_g', 'R_pl', 'T_int', 'T3', 'T2', 'T1', 'alpha', 'log_delta',
            ])
        }
        theta['R_pl'] = theta['R_pl'] * prt.nat_cst.r_jup_mean

        x = emission_spectrum(self.atmosphere, **theta, **self.constants)
        x = np.stack(x)

        return self.process(x)

    def process(self, x: Array) -> Array:
        r"""Processes spectra into network-friendly inputs"""

        if self.log_scale:
            x = np.log(x * 1e18) / np.log(1e6)

        return x


@vectorize(otypes=[Array])
def emission_spectrum(
    atmosphere: prt.Radtrans,
    CO: float,
    FeH: float,
    log_X_Fe: float,
    log_X_MgSiO3: float,
    **kwargs,
) -> Array:
    r"""Simulate emission spectrum

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L39
    """

    kwargs.update({
        'C/O': CO,
        'Fe/H': FeH,
        'log_X_cb_Fe(c)': log_X_Fe,
        'log_X_cb_MgSiO3(c)': log_X_MgSiO3,
    })

    parameters = {
        k: prm.Parameter(name=k, value=v, is_free_parameter=False)
        for k, v in kwargs.items()
    }

    _, spectrum = models.emission_model_diseq(atmosphere, parameters, AMR=True)
    return spectrum


def fixed_length_amr(p_clouds: Array, pressures: Array, scaling: int = 10, width: int = 3) -> Tuple[Array, Array]:
    r"""This function takes in the cloud base pressures for each cloud,
    and returns an array of pressures with a high resolution mesh
    in the region where the clouds are located.

    The output length is always
        len(pressures) // scaling + len(p_clouds) * (scaling - 1) * width

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L802
    """

    length = len(pressures)
    cloud_indices = np.searchsorted(pressures, np.asarray(p_clouds))

    # High resolution intervals
    def bounds(center: int, width: int) -> Tuple[int, int]:
        upper = min(center + width // 2, length)
        lower = max(upper - width, 0)
        return lower, lower + width

    intervals = [bounds(idx, scaling * width) for idx in cloud_indices]

    # Merge intervals
    while True:
        intervals, stack = sorted(intervals), []

        for interval in intervals:
            if stack and stack[-1][1] >= interval[0]:
                last = stack.pop()
                interval = bounds(
                    (last[0] + max(last[1], interval[1]) + 1) // 2,
                    last[1] - last[0] + interval[1] - interval[0],
                )

            stack.append(interval)

        if len(intervals) == len(stack):
            break
        intervals = stack

    # Intervals to indices
    indices = [np.arange(0, length, scaling)]

    for interval in intervals:
        indices.append(np.arange(*interval))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices


models.fixed_length_amr = fixed_length_amr
