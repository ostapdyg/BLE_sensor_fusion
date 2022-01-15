import numpy as np

from utilities.Multipath import Multipath
from utilities.parameters import Parameters


def signals_model(
    omega_id: float, dist: np.ndarray, ampl_coeff: np.ndarray, p: Parameters
) -> tuple[np.ndarray, np.ndarray]:
    freq_low = 2.402e9  # 2.402 GHz
    omega_low = 2.0 * np.pi * freq_low  # To radians
    omega_step = 2.0 * np.pi * 1e6  # 1 MHz in radians

    omega = omega_low + omega_id * omega_step
    hs = Multipath(omega, dist, ampl_coeff)
    delays = np.array([1.]) if p.delays.size == 0 else p.delays
    noises = np.array([0.]) if p.noises.size == 0 else p.noises

    signals = delays * (np.abs(hs)) * (np.exp(1j * 2 * np.angle(hs))) + noises

    signals = signals.conj().T
    r = signals * signals.transpose()
    return r, signals
