import numpy as np

def Multipath(omega: float, dist: np.ndarray, ampl_coeff: np.ndarray) -> float:
    c = 299792458
    multipaths = np.zeros((1, max(dist.shape)), np.csingle)
    for k in range(dist.size):
        a = c / (2 * dist[0, k] * omega)  # amplitude from distance
        phi = (dist[0, k] * omega) / c  # phase from distance
        multipaths[:, k] = ampl_coeff[0, k] * a * np.exp(-1j * phi)
    hs = np.sum(multipaths, 1)
    return hs.transpose()
