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

try:
    os.environ['pRT_input_data_path'] = os.path.expanduser('~/.cache/lampe/pRT/data')

    import petitRADTRANS as prt
    import petitRADTRANS.retrieval.models as models
    import petitRADTRANS.retrieval.parameter as prm
except Exception as e:
    print(
        f"ImportWarning: {e}. 'EES' requires",
        "  conda install -c conda-forge multinest gfortran",
        "  pip install petitRADTRANS",
    )

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

    def __init__(self, noisy: bool = True, seed: int = None, **kwargs):
        super().__init__()

        # Constants
        default = {
            'D_pl': 41.2925 * prt.nat_cst.pc,
            'pressure_scaling': 10,
            'pressure_simple': 100,
            'pressure_width': 3,
            'scale': 1e16,
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }
        self.scale = self.constants.pop('scale')

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

        # RNG
        self.noisy = noisy
        self.sigma = 1.25754e-17 * self.scale
        self.rng = np.random.default_rng(seed)

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
        x = self.process(x)

        if self.noisy:
            x = x + self.sigma * self.rng.standard_normal(x.shape)

        return x

    def process(self, x: Array) -> Array:
        r"""Processes spectra into network-friendly inputs"""

        return x * self.scale


@vectorize(otypes=[Array])
def emission_spectrum(
    atmosphere: prt.Radtrans,
    CO: float,
    FeH: float,
    log_X_Fe: float,
    log_X_MgSiO3: float,
    **kwargs,
) -> Array:
    r"""Simulates emission spectrum

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L41
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


@vectorize(signature='(m),(n)->(n)')
def pt_profile(theta: Array, pressures: Array) -> Array:
    r"""Calculates the pressure-temperature profile

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L639
    """

    CO, FeH, *_, T_int, T3, T2, T1, alpha, log_delta = theta

    T3 = ((3 / 4 * T_int ** 4 * (0.1 + 2 / 3)) ** (1 / 4)) * (1 - T3)
    T2 = T3 * (1 - T2)
    T1 = T2 * (1 - T1)
    delta = (1e6 * 10 ** (-3 + 5 * log_delta)) ** (-alpha)

    return models.PT_ret_model(
        np.array([T1, T2, T3]),
        delta,
        alpha,
        T_int,
        pressures,
        FeH,
        CO,
    )
