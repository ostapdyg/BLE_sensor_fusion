import numpy as np
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from interpolation.wsinterp import wsinterp

sklearn_interpolations = [
    "linear",
    "nearest",
    # "nearest-up",
    # "zero",
    # "slinear",
    "quadratic",
    "cubic",
    "previous",
    "next",
]


def interpolate(signals: np.ndarray, kind: str = "Whittaker–Shannon"):
    interp_signal = []

    x = np.arange(signals.shape[1])
    if kind in sklearn_interpolations:
        for signal in signals:
            idx = np.where(~np.isnan(signal))[0]
            interp_signal.append(
                interp1d(x[idx], signal[idx], kind=kind, bounds_error=False)(x)
            )
    elif kind == "Whittaker–Shannon":
        for signal in tqdm(signals):
            idx = np.where(~np.isnan(signal))[0]
            interp_signal.append(wsinterp(x, x[idx], signal[idx]))

    return np.array(interp_signal)
