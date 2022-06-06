import numpy as np
import scipy

from .simul import FREQS, C_SPEED
# Returns fft_data, fft_freqs
def fft(vals, ts, n_mult = np.pi*10):
    sample_rate = ts.shape[-1]/(ts[-1] - ts[0])
    n = int(vals.shape[-1] * n_mult)
    # return np.abs(np.fft.fft(vals, n=n)), np.fft.fftfreq(n)*sample_rate
    res = np.rec.fromarrays([np.fft.fftfreq(n)*sample_rate, np.abs(np.fft.fft(vals, n=n))]) # Fix shape
    res.sort()
    return res.f1, res.f0


def interp_lsq_spline(x, xs, y, k=3):

    spl_r = scipy.interpolate.splrep(x, y.real, w=np.ones(x.shape)/0.0001, k=k)
    spl_i = scipy.interpolate.splrep(x, y.imag, w=np.ones(x.shape)/0.0001, k=k)
    return scipy.interpolate.splev(xs, spl_r)+1j*scipy.interpolate.splev(xs, spl_i)


def estimate_distance(signal_v, freqs=FREQS):

    signal_interp = interp_lsq_spline(freqs[~np.isnan(signal_v)], freqs, signal_v[~np.isnan(signal_v)])
    fft_vals, fft_freqs = fft(signal_interp, freqs)
    i_fmax = np.argmax(np.abs(fft_vals))
    return np.abs(fft_freqs[i_fmax]*C_SPEED)

