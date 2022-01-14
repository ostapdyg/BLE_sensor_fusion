import numpy as np

from .Multipath import Multipath


def ctranspose(arr: np.ndarray) -> np.ndarray:
    # Explanation of the math involved:
    # x      == Real(X) + j*Imag(X)
    # conj_x == Real(X) - j*Imag(X)
    # conj_x == Real(X) + j*Imag(X) - 2j*Imag(X) == x - 2j*Imag(X)
    tmp = arr.transpose()
    return tmp - 2j * tmp.imag


def signals_model(
    omega_id: float,
    dist: np.ndarray,
    ampl_coeff: np.ndarray,
    delays: np.ndarray,
    noises: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # freq_low = 2.402e9; #2.402 GHz
    freq_low = 2.402e9  # 2.402 GHz
    # freq_high = 2.480e9; #2.480 GHz
    # omega_low = 2.0 * np.pi * freq_low; # To radians
    # omega_step = 2.0 * np.pi * 1e6; # 1 MHz in radians
    # freq_high = 2.480e9  # 2.480 GHz
    omega_low = 2.0 * np.pi * freq_low  # To radians
    omega_step = 2.0 * np.pi * 1e6  # 1 MHz in radians

    # omega = omega_low + omega_id * omega_step;
    # hs = Multipath(omega, dist, ampl_coeff);
    omega = omega_low + omega_id * omega_step
    hs = Multipath(np.array([omega]), dist, ampl_coeff)
    # if isempty(delays)
    #     delays = 1;
    # end
    # if isempty(noises)
    #     noises = 0;
    # end
    if not max(delays.shape):
        delays = np.array([1])
    if not max(noises.shape):
        noises = np.array([0])
    # print(f"    signals_model delays:{delays.shape}")
    # print(f"    signals_model hs:{hs.shape}")

    # signals = delays .* abs(hs) .* exp(1i * 2 * angle(hs)) + noises;
    signals = delays * (np.abs(hs)) * (np.exp(1j * 2 * np.angle(hs))) + noises

    # signals = signals.';
    # r = signals*signals';
    signals = ctranspose(signals)
    r = signals * signals.transpose()
    # print(f"    signals_model signals:{signals}")
    # print(f"    signals_model r:{r}")
    return (r, signals)


# end
