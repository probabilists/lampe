r"""Hodgkin-Huxley (HH) simulator"""

import numba as nb
import numpy as np
import scipy.stats as sp
import torch

from numpy import ndarray as Array
from torch import Tensor, BoolTensor

from . import Simulator
from .priors import Distribution, JointUniform
from ..utils import jit


class HH(Simulator):
    r"""Hodgkin-Huxley"""

    def __init__(self, summary: bool = True, **kwargs):
        super().__init__()

        self.summary = summary

        # Constants
        default = {
            'duration': 80.,  # s
            'time_step': 0.02,  # s
            'padding': 10.,  # s
            'initial_voltage': -70., # mV
            'current': 5e-4 / (np.pi * 7e-3 ** 2),  # uA / cm^2
        }

        self.constants = nb.typed.Dict()
        for k, v in default.items():
            self.constants[k] = kwargs.get(k, v)

        # Prior
        bounds = torch.tensor([
            [0.5, 80.],     # mS/cm^2
            [1e-4, 15.],    # mS/cm^2
            [1e-4, .6],     # mS/cm^2
            [1e-4, .6],     # mS/cm^2
            [50., 3000.],   # ms
            [-90., -40.],   # mV
            [1e-4, .15],    # uA/cm^2
            [-100., -35.],  # mV
        ])

        self.register_buffer('low', bounds[:, 0])
        self.register_buffer('high', bounds[:, 1])

    def marginal_prior(self, mask: BoolTensor) -> Distribution:
        r""" p(theta_a) """

        return JointUniform(self.low[mask], self.high[mask])

    def labels(self) -> list[str]:
        labels = [
            r'g_{\mathrm{Na}}', r'g_{\mathrm{K}}', r'g_{\mathrm{M}}', 'g_l',
            r'\tau_{\max}', 'V_t', r'\sigma', 'E_l',
        ]
        labels = [f'${l}$' for l in labels]

        return labels

    def sample(self, theta: Tensor, shape: torch.Size = ()) -> Tensor:
        r""" x ~ p(x | theta) """

        _theta = theta.cpu().numpy().astype(np.float64)
        _theta = np.broadcast_to(_theta, shape + _theta.shape)

        x = voltage_trace(_theta, self.constants)

        if self.summary:
            x = summarize(x, self.constants)

        x = torch.from_numpy(x).to(theta.device)

        return x


@jit
def voltage_trace(theta: Array, constants: dict) -> Array:
    r"""Simulate Hodgkin-Huxley voltage trace

    References:
        https://github.com/mackelab/sbi/blob/main/examples/HH_helper_functions.py
    """

    # Parameters
    T = constants['duration']
    dt = constants['time_step']
    pad = constants['padding']
    V_0 = constants['initial_voltage']
    I = constants['current']

    theta = theta.reshape((1,) + theta.shape)
    g_Na, g_K, g_M, g_leak, tau_max, V_t, sigma, E_leak = [
        theta[..., i] for i in range(8)
    ]

    C = 1.  # uF/cm^2
    E_Na = 53.  # mV
    E_K = -107.  # mV

    # Kinetics
    exp = np.exp
    efun = lambda x: np.where(
        np.abs(x) < 1e-4,
        1 - x / 2,
        x / (exp(x) - 1)
    )

    alpha_n = lambda x: 0.032 * efun(-0.2 * (x - 15.)) / 0.2
    beta_n = lambda x: 0.5 * exp(-(x - 10.) / 40.)
    tau_n = lambda x: 1 / (alpha_n(x) + beta_n(x))
    n_inf = lambda x: alpha_n(x) / (alpha_n(x) + beta_n(x))

    alpha_m = lambda x: 0.32 * efun(-0.25 * (x - 13.)) / 0.25
    beta_m = lambda x: 0.28 * efun(0.2 * (x - 40.)) / 0.2
    tau_m = lambda x: 1 / (alpha_m(x) + beta_m(x))
    m_inf = lambda x: alpha_m(x) / (alpha_m(x) + beta_m(x))

    alpha_h = lambda x: 0.128 * exp(-(x - 17.) / 18)
    beta_h = lambda x: 4. / (1. + exp(-0.2 * (x - 40.)))
    tau_h = lambda x: 1 / (alpha_h(x) + beta_h(x))
    h_inf = lambda x: alpha_h(x) / (alpha_h(x) + beta_h(x))

    tau_p = lambda x: tau_max / (3.3 * exp(0.05 * (x + 35.)) + exp(-0.05 * (x + 35.)))
    p_inf = lambda x: 1. / (1. + exp(-0.1 * (x + 35.)))

    # Iterations
    timesteps = np.arange(0, T, dt)
    voltages = np.empty(V_t.shape + timesteps.shape)

    V = np.full_like(V_t, V_0)
    V_rel = V - V_t

    n = n_inf(V_rel)
    m = m_inf(V_rel)
    h = h_inf(V_rel)
    p = p_inf(V)

    for i, t in enumerate(timesteps):
        tau_V = C / (
            g_Na * m ** 3 * h
            + g_K * n ** 4
            + g_M * p
            + g_leak
        )

        V_inf = tau_V * (
            E_Na * g_Na * m ** 3 * h
            + E_K * g_K * n ** 4
            + E_K * g_M * p
            + E_leak * g_leak
            + I * (pad <= t < T - pad)
            + sigma * np.random.randn(*V.shape) / dt ** 0.5  # noise
        ) / C

        V = V_inf + (V - V_inf) * exp(-dt / tau_V)
        V_rel = V - V_t

        n = n_inf(V_rel) + (n - n_inf(V_rel)) * exp(-dt / tau_n(V_rel))
        m = m_inf(V_rel) + (m - m_inf(V_rel)) * exp(-dt / tau_m(V_rel))
        h = h_inf(V_rel) + (h - h_inf(V_rel)) * exp(-dt / tau_h(V_rel))
        p = p_inf(V) + (p - p_inf(V)) * exp(-dt / tau_p(V))

        voltages[..., i] = V

    return voltages.reshape(voltages.shape[1:])


def summarize(x: Array, constants: dict) -> Array:
    r"""Compute summary statistics"""

    T = constants['duration']
    dt = constants['time_step']
    pad = constants['padding']

    t = np.arange(0, T, dt)

    # Number of spikes
    spikes = np.maximum(x, -10)
    spikes = np.diff(np.sign(np.diff(spikes)))
    spikes = np.sum(spikes < 0, axis=-1)

    # Resting moments
    rest = x[..., (pad / 2 <= t) * (t < pad)]
    rest_mean = np.mean(rest, axis=-1)
    rest_std = np.std(rest, axis=-1)

    # Moments
    x = x[..., (pad <= t) * (t < T - pad)]
    x_mean = np.mean(x, axis=-1)
    x_var = np.var(x, axis=-1)
    x_skew = sp.skew(x, axis=-1)
    x_kurtosis = sp.kurtosis(x, axis=-1)

    return np.stack([
        spikes,
        rest_mean, rest_std,
        x_mean, x_var, x_skew, x_kurtosis,
    ], axis=-1)
