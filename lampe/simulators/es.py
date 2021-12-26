r"""Exoplanet Spectra (ES) simulator"""

import numpy as np
import os
import torch

try:
    os.environ['pRT_input_data_path'] = os.path.expanduser('~/.cache/lampe/pRT/data')

    import petitRADTRANS as prt
    import petitRADTRANS.retrieval.models as models

    from petitRADTRANS.retrieval.parameter import Parameter
except Exception as e:
    print('Error while importing required modules for exoplanet spectra analysis.')
    print('Requires')
    print('  conda install -c conda-forge multinest gfortran')
    print('  pip install petitRADTRANS')
    raise

from numpy import ndarray as Array
from torch import Tensor, BoolTensor

from . import Simulator
from .priors import Distribution, JointUniform
from ..utils import disk_cache, vectorize


class ES(Simulator):
    r"""Exoplanet Spectra

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/
    """

    def __init__(self, **kwargs):
        super().__init__()

        # Constants
        default = {
            'D_pl': 41.2925 * prt.nat_cst.pc,
            'pressure_scaling': 10,
            'pressure_width': 3,
            'pressure_simple': 100,
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        self.atmosphere = disk_cache(prt.Radtrans)(
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
            self.constants['pressure_width'] * (self.constants['pressure_scaling'] - 1)
        )

        self.atmosphere.setup_opa_structure(np.logspace(-6, 2, levels))

        # Prior
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
            [0.9, 2.],      # R_P [R_Jupyter]
            [300., 2300.],  # T_0 [K]
            [0., 1.],       # ∝ T_3 / T_connect
            [0., 1.],       # ∝ T_2 / T_3
            [0., 1.],       # ∝ T_1 / T_2
            [1., 2.],       # alpha
            [0., 1.],       # ∝ log delta / alpha
        ])

        self.register_buffer('low', bounds[:, 0])
        self.register_buffer('high', bounds[:, 1])

    def marginal_prior(self, mask: BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return JointUniform(self.low[mask], self.high[mask])

    def labels(self) -> list[str]:
        labels = [
            r'{\rm C/O}', r'\left[{\rm Fe/H}\right]', r'\log P_{\rm quench}', r'\log X_{\rm Fe}', r'\log X_{\rm MgDiO_3}',
            r'f_{\rm sed}', r'\log K_{zz}', r'\sigma_g', r'\log g', r'R_P',
            r'T_0', r'\frac{T_3}{T_{\rm connect}}', r'\frac{T_2}{T_3}', r'\frac{T_1}{T_2}',
            r'\alpha', r'\frac{\log \delta}{\alpha}',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def sample(self, theta: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" x ~ p(x | theta) """

        _theta = theta.cpu().numpy().astype(np.float64)
        _theta = np.broadcast_to(_theta, shape + _theta.shape)

        x = diseq_chemistry_emission(
            _theta,
            self.atmosphere,
            self.constants,
        )

        x = self.process(x)
        x = x.to(theta.device)

        return x

    def process(self, x: Array) -> Tensor:
        r"""Process spectra into network-ready inputs"""

        x = np.log(x) - np.log(1e-18)
        x = x / np.log(1e6)
        x = x.astype(np.float32)

        x = torch.from_numpy(x)

        return x


class FixedParameter(Parameter):
    r"""Fixed parameter"""

    def __init__(self, name: str, value: float):
        super().__init__(name=name, value=value, is_free_parameter=False)


def diseq_chemistry_emission(theta: Array, atmosphere: prt.Radtrans, constants: dict) -> Array:
    r"""Simulate Disequilibrium Chemistry Emission(s)

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L39
    """

    kwargs = {
        k: theta[..., i]
        for i, k in enumerate([
            'CO', 'FeH', 'log_pquench', 'log_X_Fe', 'log_X_MgSiO3', 'fsed', 'log_kzz',
            'sigma_lnorm', 'log_g', 'R_pl', 'T_int', 'T3', 'T2', 'T1', 'alpha', 'log_delta',
        ])
    }
    kwargs['R_pl'] = kwargs['R_pl'] * prt.nat_cst.r_jup_mean

    @vectorize(otypes=[Array])
    def temporary(CO: float, FeH: float, log_X_Fe: float, log_X_MgSiO3: float, **kwargs) -> Array:
        parameters = {
            k: FixedParameter(k, v)
            for k, v in kwargs.items()
        }

        parameters['C/O'] = FixedParameter('C/O', CO)
        parameters['Fe/H'] = FixedParameter('C/O', FeH)
        parameters['log_X_cb_Fe(c)'] = FixedParameter('log_X_cb_Fe(c)', log_X_Fe)
        parameters['log_X_cb_MgSiO3(c)'] = FixedParameter('log_X_cb_MgSiO3(c)', log_X_MgSiO3)

        _, spectra = models.emission_model_diseq(atmosphere, parameters, AMR=True)
        return spectra

    spectra = temporary(**kwargs, **constants)
    spectra = np.stack(spectra)

    return spectra


def fixed_length_amr(p_clouds: Array, pressures: Array, scaling: int = 10, width: int = 3) -> tuple[Array, Array]:
    r"""This function takes in the cloud base pressures for each cloud,
    and returns an array of pressures with a high resolution mesh
    in the region where the clouds are located.

    The output length is always
        len(pressures[::scaling]) + len(p_clouds) * width * (scaling - 1)

    References:
        https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/retrieval/models.py#L802
    """

    length = len(pressures)
    cloud_indices = np.searchsorted(pressures, np.asarray(p_clouds))

    # High resolution intervals
    intervals = []

    for idx in cloud_indices:
        lower = max(idx - scaling * width // 2, 0)
        upper = min(lower + scaling * width, length)
        lower = min(upper - scaling * width, lower)

        intervals.append((lower, upper))

    # Merge intervals
    merged = False
    while not merged:
        intervals = sorted(intervals)
        stack = []

        for interval in intervals:
            if stack and stack[-1][1] >= interval[0]:
                last = stack.pop()
                total_width = last[1] - last[0] + interval[1] - interval[0]
                lower, upper = last[0], max(last[1], interval[1])

                ## Keep constant width
                left = False
                while upper - lower < total_width:
                    left = not left
                    if left and lower > 0:
                        lower = lower - 1
                    elif not left and upper < length:
                        upper = upper + 1

                stack.append((lower, upper))
            else:
                stack.append(interval)

        if len(intervals) == len(stack):
            merged = True
        intervals = stack

    # Intervals to indices
    indices = [np.arange(0, length, scaling)]

    for interval in intervals:
        indices.append(np.arange(*interval))

    indices = np.unique(np.concatenate(indices))

    return pressures[indices], indices


models.fixed_length_amr = fixed_length_amr
