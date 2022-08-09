import numpy as np
import scipy
from scipy.interpolate import interp1d

BASE_FREQ = 2.4e9 #Hz
FREQ_STEP = 2e6 #Hz
FREQ_NUM = 80

FREQS = np.arange(BASE_FREQ, BASE_FREQ+FREQ_STEP*FREQ_NUM-1, FREQ_STEP)
C_SPEED = 299_792_458

SIGNAL_STD = 0.0

# Returns fft_data, fft_freqs
def fft(vals, ts, n_mult = 1000):
    sample_rate = ts.shape[-1]/(ts[-1] - ts[0])
    n = int(vals.shape[-1] * n_mult)
    # return np.abs(np.fft.fft(vals, n=n)), np.fft.fftfreq(n)*sample_rate
    res = np.rec.fromarrays([np.fft.fftfreq(n)*sample_rate, np.abs(np.fft.fft(vals, n=n))]) # Fix shape
    res.sort()
    return res.f1, res.f0


# def interp_lsq_spline(x, xs, y, k=3):

#     spl_r = scipy.interpolate.splrep(x, y.real, w=np.ones(x.shape)/0.0001, k=k)
#     spl_i = scipy.interpolate.splrep(x, y.imag, w=np.ones(x.shape)/0.0001, k=k)
#     return scipy.interpolate.splev(xs, spl_r)+1j*scipy.interpolate.splev(xs, spl_i)



def estimate_distance(signal_v, freqs=FREQS):

    # signal_interp = interp_lsq_spline(freqs[~np.isnan(signal_v)], freqs, signal_v[~np.isnan(signal_v)])
    signal_interp = signal_v
    fft_vals, fft_freqs = fft(signal_interp, freqs)
    i_fmax = np.argmax(np.abs(fft_vals))
    return np.abs(fft_freqs[i_fmax]*C_SPEED)
    # *(1+np.random.normal(0, 0.2))

def simul_signals_shift_full(anchor, target)->np.ndarray:
    virt_targets = [
        # (target, 0.2),
        (target, 1),
        # (np.array([target[0], 10 - target[1], target[2]]), 0.4),
        # (np.array([target[0],  - target[1], target[2]]), 0.4),
        # (np.array([12 - target[0], target[1], target[2]]), 0.4),
        # (np.array([-target[0], target[1], target[2]]), 0.4),
        # (np.array([target[0], target[1], -target[2]]), 0.4),
        ]
    # if((anchor[0] == 0) and (anchor[1] == 0)):
    #     virt_targets[0] = (target, 0.2),
    signal_v = np.zeros(FREQ_NUM, dtype=np.complex128)
    for virt_target, reflection_coeff in virt_targets:
        path_vector = virt_target - anchor
        path_len = np.sqrt(path_vector.dot(path_vector))

        omega_v = 2*np.pi*FREQS
        a_v = C_SPEED / (2 * path_len * omega_v)  # amplitude from distance
        # a_v = 1  # amplitude from distance

        phi_v = (path_len * omega_v) / C_SPEED  # phase from distance
        path_signal = reflection_coeff * a_v * (np.exp(-1j * phi_v) + np.random.normal(0, SIGNAL_STD)+1j*np.random.normal(0, SIGNAL_STD))

        signal_v += path_signal

    return signal_v

    pass


# dists: 0:10, 0.01
def estimate_dist_probf(signal_v, freqs=FREQS, dist_max=10):
    signal_interp = signal_v
    fft_vals, fft_freqs = fft(signal_interp, freqs)
    # dists = np.arange(0, dist_max*100)
    # probs = np.zeros([dist_max*100])
    res = interp1d(fft_freqs*C_SPEED, fft_vals)
    return lambda x:res(x)
    # return fft_vals, fft_freqs*C_SPEED