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
def randomized_sinc_interp(x:np.ndarray, xp:np.ndarray, fp:np.ndarray, sigma_coeff=0.8, left=None, right=None)->np.ndarray:

    Tn = (xp[-1] - xp[0])/xp.shape[0]
#     print(xp.shape, xp[0], xp[1], Tn)
    xp_regular = np.arange(xp[0], xp[0] + Tn*xp.shape[0], Tn)
    
    xp_deltas = xp - xp_regular

    xp_result = xp_regular + xp_deltas * sigma_coeff 

    # shape = (nxp, nx), nxp copies of x data span axis 1
    u = np.resize(x, (len(xp), len(x)))
    # Must take transpose of u for proper broadcasting with xp.
    # shape = (nx, nxp), v(xp) data spans axis 1
    # v = (xp - u.T) / (Tn)
    v = (xp_result - u.T) / (Tn)
    # shape = (nx, nxp), m(v) data spans axis 1
    m =   fp * np.sinc(v)
    # Sum over m(v) (axis 1)
    fp_at_x = np.sum(m, axis=1)

    # Enforce left and right
    if left is None:
        left = fp[0]
    fp_at_x[x < xp[0]] = left
    if right is None:
        right = fp[-1]
    fp_at_x[x > xp[-1]] = right

    return fp_at_x


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
    elif kind == "randomized_sinc":
        for signal in tqdm(signals):
            idx = np.where(~np.isnan(signal))[0]
            interp_signal.append(randomized_sinc_interp(x, x[idx], signal[idx]))

    return np.array(interp_signal)


