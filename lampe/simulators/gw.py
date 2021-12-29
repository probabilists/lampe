r"""Gravitational Waves (GW) simulator"""

import numpy as np
import os
import torch

try:
    os.environ['GWPY_RCPARAMS'] = '0'

    from gwpy.timeseries import TimeSeries
    from lal import MSUN_SI
    from lalsimulation import SimInspiralTransformPrecessingNewInitialConditions
    from pycbc.catalog import Merger
    from pycbc.detector import Detector
    from pycbc.psd import welch
    from pycbc.waveform import get_fd_waveform
except Exception as e:
    print('Error while importing required modules for gravitational wave analysis.')
    print('Requires')
    print('  pip install gwpy pycbc')
    raise

from numpy import ndarray as Array
from torch import Tensor, BoolTensor

from . import Simulator
from .priors import (
    Distribution,
    Joint,
    Uniform,
    SineUniform,
    CosineUniform,
    Sort,
    Maximum,
    Minimum,
)
from ..utils import cache, disk_cache, vectorize


class GW(Simulator):
    r"""Gravitational Waves

    References:
        https://github.com/stephengreen/lfi-gw
    """

    def __init__(
        self,
        n_components: int = 2 ** 7,  # 128
        reduced_basis: bool = True,
        noisy: bool = True,
        **kwargs,
    ):
        super().__init__()

        self._shortcut = False
        self.noisy = noisy

        # Constants
        default = {
            'event': 'GW150914',
            'detectors': ('H1', 'L1'),
            'approximant': 'IMRPhenomPv2',
            'duration': 4.,  # s
            'buffer': 2.,  # s
            'segment': 1024.,  # s
            'sample_rate': 1024.,  # Hz
            'f_ref': 40.,  # Hz
            'f_lower': 20.,  # Hz
        }

        self.constants = {
            k: kwargs.get(k, v)
            for k, v in default.items()
        }

        self.event_dft = crop_dft(event_dft(**self.constants), **self.constants)
        self.event_nsd = crop_dft(event_nsd(**self.constants), **self.constants)

        # Reduced SVD basis
        if reduced_basis:
            self.basis = svd_basis(
                n_components,
                **self.constants,
            ).astype(np.complex64)
        else:
            self.basis = None

        # Prior
        bounds = torch.tensor([
            [10., 80.],               # primary mass [solar masses]
            [10., 80.],               # secondary mass [solar masses]
            [0., 2 * np.pi],          # coalesence phase [rad]
            [-0.1, 0.1],              # coalescence time [s]
            [100., 1000.],            # luminosity distance [megaparsec]
            [0., 0.88],               # a_1 [/]
            [0., 0.88],               # a_2 [/]
            [0., np.pi],              # tilt_1 [rad]
            [0., np.pi],              # tilt_2 [rad]
            [0., 2 * np.pi],          # phi_12 [rad]
            [0., 2 * np.pi],          # phi_jl [rad]
            [0., np.pi],              # theta_jn [rad]
            [0., np.pi],              # polarization [rad]
            [0., 2 * np.pi],          # right ascension [rad]
            [-np.pi / 2, np.pi / 2],  # declination [rad]
        ])

        self.register_buffer('lower', bounds[:, 0])
        self.register_buffer('upper', bounds[:, 1])

    def marginal_prior(self, mask: BoolTensor) -> Distribution:
        r""" p(theta_a) """

        if mask is ...:
            mask = [True] * len(self.lower)

        marginals = []

        if mask[0] or mask[1]:
            base = Uniform(self.lower[0], self.upper[0])

            if mask[0] and mask[1]:
                law = Sort(base, n=2, descending=True)
            elif mask[0]:
                law = Maximum(base, n=2)
            elif mask[1]:
                law = Minimum(base, n=2)

            marginals.append(law)

        for i, b in enumerate(mask[2:], start=2):
            if not b:
                continue

            if i in [7, 8, 11]:  # [tilt_1, tilt_2, theta_jn]
                m = CosineUniform(self.lower[i], self.upper[i])
            elif i == 14:  # declination
                m = SineUniform(self.lower[i], self.upper[i])
            else:
                m = Uniform(self.lower[i], self.upper[i])

            marginals.append(m)

        return Joint(marginals)

    def labels(self) -> list[str]:
        labels = [
            'm_1', 'm_2', r'\phi_c', 't_c', 'd_L',
            'a_1', 'a_2', r'\theta_1', r'\theta_2', r'\phi_{12}', r'\phi_{JL}',
            r'\theta_{JN}', r'\psi', r'\alpha', r'\delta',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def sample(self, theta: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" x ~ p(x | theta) """

        _theta = theta.cpu().numpy().astype(float)
        x = gravitational_waveform(_theta, **self.constants)
        x = crop_dft(x, **self.constants)
        x = whiten_dft(x, self.event_nsd)

        if self._shortcut:
            return x

        x = self.process(x)
        x = x.to(theta.device)
        x = torch.broadcast_to(x, shape + x.shape)

        if self.noisy:
            x = self.noise(x)

        return x

    def process(self, x: Array) -> Tensor:
        r"""Process waveforms into network-ready inputs"""

        x = x.astype(np.complex64)

        if self.basis is not None:
            x = x @ self.basis

        x = torch.from_numpy(x)
        x = torch.view_as_real(x)

        return x

    def noise(self, x: Tensor) -> Tensor:
        r"""Add unit Gaussian noise"""

        return x + torch.randn(*x.shape)

    @property
    def event(self) -> Tuple[Tensor, Tensor]:
        r"""Event of reference"""

        theta = None
        x = self.process(self.event_dft)

        return theta, x


@cache
def ligo_detector(name: str):
    r"""Fetch LIGO detector"""

    return Detector(name)


@cache
def event_gps(event: str = 'GW150914') -> float:
    r"""Fetch event's GPS time"""

    return Merger(event).data['GPS']


@cache
def tukey_window(
    duration: int,  # s
    sample_rate: float,  # Hz
    roll_off: float = 0.4,  # s
) -> Array:
    r"""Tukey window function

    References:
        https://en.wikipedia.org/wiki/Window_function
    """

    from scipy.signal import tukey

    length = int(duration * sample_rate)
    alpha = 2 * roll_off / duration

    return tukey(length, alpha)


@disk_cache
def event_nsd(
    event: str,
    detectors: tuple[str, ...],
    duration: float,  # s
    segment: float,  # s
    **kwargs,
) -> Array:
    r"""Fetch event's Noise Spectral Density (NSD)

    Wikipedia:
        https://en.wikipedia.org/wiki/Noise_spectral_density

    References:
        https://github.com/gwastro/pycbc/blob/master/pycbc/noise/gaussian.py#L35
    """

    time = event_gps(event) - duration

    # Fetch
    nsds = []

    for det in detectors:
        strain = TimeSeries.fetch_open_data(det, time - segment, time, cache=True).to_pycbc(copy=False)

        win = tukey_window(duration, strain.sample_rate)
        win_factor = np.sum(win ** 2) / len(win)
        psd = welch(strain, len(win), len(win), window=win, avg_method='median') * win_factor

        nsds.append(0.5 * np.sqrt(psd.data / psd.delta_f))

    return np.stack(nsds)


@disk_cache
def event_dft(
    event: str,
    detectors: tuple[str, ...],
    duration: float,  # s
    buffer: float,  # s
    **kwargs,
) -> Array:
    r"""Fetch event's Discrete Fourier Transform (DFT)"""

    time = event_gps(event) + buffer

    # Fetch
    dfts = []

    for det in detectors:
        strain = TimeSeries.fetch_open_data(det, time - duration, time, cache=True).to_pycbc(copy=False)

        win = tukey_window(duration, strain.sample_rate)
        dft = (strain * win).to_frequencyseries().cyclic_time_shift(buffer)

        dfts.append(dft.data)

    return np.stack(dfts)


@vectorize(otypes=[float] * 7)
def lal_spins(*args) -> tuple[float, ...]:
    r"""Convert LALInference geometric parameters to LALSimulation spins"""

    return tuple(SimInspiralTransformPrecessingNewInitialConditions(*args))


@vectorize(otypes=[Array, Array])
def plus_cross(**kwargs) -> tuple[Array, Array]:
    r"""Simulate frequency-domain plus and cross polarizations
    of gravitational wave
    """

    hp, hc = get_fd_waveform(**kwargs)
    return hp.numpy(), hc.numpy()


def gravitational_waveform(
    theta: Array,
    event: str,
    detectors: tuple[str, ...],
    approximant: str,
    duration: float,  # s
    sample_rate: float,  # Hz
    f_ref: float,  # Hz
    f_lower: float,  # Hz
    **kwargs,
) -> Array:
    r"""Simulate a frequency-domain gravitational wave projected onto LIGO detectors

    References:
        http://pycbc.org/pycbc/latest/html/waveform.html
        http://pycbc.org/pycbc/latest/html/detector.html
    """

    # Parameters
    m_1, m_2, phi_c, t_c, d_L, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, theta_jn, psi, alpha, delta = [
        theta[..., i] for i in range(15)
    ]

    iota, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = lal_spins(
        theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
        m_1 * MSUN_SI, m_2 * MSUN_SI, f_ref, phi_c
    )

    # Gravitational wave
    hp, hc = plus_cross(
        ## Constants
        approximant=approximant,
        delta_f=1/duration,
        f_ref=f_ref,
        f_lower=f_lower,
        f_final=sample_rate/2,
        ## Variables
        mass1=m_1,
        mass2=m_2,
        coa_phase=phi_c,
        distance=d_L,
        inclination=iota,
        spin1x=spin1x,
        spin2x=spin2x,
        spin1y=spin1y,
        spin2y=spin2y,
        spin1z=spin1z,
        spin2z=spin2z,
    )

    hp, hc = np.stack(hp), np.stack(hc)

    # Projection on detectors
    time = event_gps(event)
    angular_speeds = -1j * 2 * np.pi * np.arange(hp.shape[-1]) / duration

    strains = []

    for name in detectors:
        det = ligo_detector(name)

        ## Noiseless strain
        fp, fc = det.antenna_pattern(alpha, delta, psi, time)
        s = fp[..., None] * hp + fc[..., None] * hc

        ## Cyclic time shift
        dt = det.time_delay_from_earth_center(alpha, delta, time) + t_c
        s = s * np.exp(dt[..., None] * angular_speeds)

        strains.append(s)

    return np.stack(strains, axis=-2)


def crop_dft(
    dft: Array,
    duration: float,  # s
    sample_rate: float,  # Hz
    f_lower: float,  # Hz
    **kwargs,
) -> Array:
    r"""Crop low and high frequencies of Discrete Fourier Transform (DFT)"""

    return dft[..., int(duration * f_lower):int(duration * sample_rate / 2) + 1]


def whiten_dft(dft: Array, nsd: Array) -> np.ndarray:
    r"""Whiten Discrete Fourier Transform (DFT) w.r.t. Noise Spectral Density (NSD)"""

    return dft / nsd


@store
def svd_basis(
    n_components: int,
    n_samples: int = 2 ** 15,
    batch_size: int = 2 ** 10,
    seed: int = 0,
    **kwargs,
) -> Array:
    r"""Build Singular Value Decompostition (SVD) basis"""

    from tqdm import tqdm

    sim = GW(reduced_basis=False, noisy=False, **kwargs)
    sim.high[4] = sim.low[4]
    sim._shortcut = True

    print('Generating samples...')

    xs = []

    for _ in tqdm(range(n_samples // batch_size), unit_scale=batch_size):
        _, x = sim.joint((batch_size,))
        xs.append(x)

    x = np.stack(xs).reshape(-1, x.shape[-1])

    print('Computing SVD basis...')

    try:
        from sklearn.utils.extmath import randomized_svd

        _, _, Vh = randomized_svd(
            x,
            n_components=n_components,
            n_oversamples=n_components,
            random_state=seed,
        )
    except ImportError as e:
        _, _, Vh = np.linalg.svd(x, full_matrices=False)
        Vh = Vh[:n_components]

    V = Vh.T.conj()

    print('Done!')

    return V
