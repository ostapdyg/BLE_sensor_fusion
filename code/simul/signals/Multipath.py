import numpy as np

def Multipath(omega: float or np.ndarray, dist: np.ndarray, ampl_coeff: np.ndarray) -> float:
    c = 299792458
    omega = np.atleast_1d(omega)    #i scalar, convert to array
    multipaths = np.zeros((1, max(dist.shape), omega.shape[0]), np.csingle)
    for k in range(dist.size):
        a = c / (2 * dist[0, k] * omega)  # amplitude from distance
        phi = (dist[0, k] * omega) / c  # phase from distance
        multipaths[:, k] = ampl_coeff[0, k] * a * np.exp(-1j * phi)
    hs = np.sum(multipaths, 1)
    return hs.transpose()

if(__name__ == "__main__"):
    dist = np.expand_dims(np.array([1,2,3,4]), axis=0)
    ampl_coeff = np.expand_dims(
        np.array([0.2, 0.0, 0.0, 0.6]).repeat(dist.shape[0]), axis=0
    )
    print(Multipath(np.array([2.402e9, 2.404e9, 2.406e9]), dist, ampl_coeff))