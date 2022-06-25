import numpy as np

import matplotlib.ticker as mticker

import matplotlib.pyplot as plt
from Src.simul import simul_get_pos
from Src.simul import  C_SPEED, FREQ_NUM

from Src.plot import plot_format_axes


BASE_FREQ = 2.4e9 #Hz
FREQ_STEP = 2e6 #Hz
FREQ_NUM = 160

FREQS = np.arange(BASE_FREQ, BASE_FREQ+FREQ_STEP*FREQ_NUM-1, FREQ_STEP)
# def estimate_distance(signal_v, freqs=FREQS):

#     # signal_interp = interp_lsq_spline(freqs[~np.isnan(signal_v)], freqs, signal_v[~np.isnan(signal_v)])
#     fft_vals, fft_freqs = fft(signal_interp, freqs)
#     # i_fmax = np.argmax(np.abs(fft_vals))
#     # return np.abs(fft_freqs[i_fmax]*C_SPEED)

def fft(vals, ts, n_mult = np.pi*10):
    sample_rate = ts.shape[-1]/(ts[-1] - ts[0])
    # n = int(vals.shape[-1] * n_mult)
    n = 1000
    # return np.abs(np.fft.fft(vals, n=n)), np.fft.fftfreq(n)*sample_rate
    res = np.rec.fromarrays([np.fft.fftfreq(n)*sample_rate, np.abs(np.fft.fft(vals, n=n))]) # Fix shape
    res.sort()
    # print(res.f0[-1])
    return res.f1, res.f0


def my_correlation_use(stear_vects, iqs) -> np.ndarray:

    return np.abs(((stear_vects.T @ iqs) * stear_vects.conj().T).sum(1))


def calc_stear_vect(freqs, dists) -> np.ndarray:
    """calculate ideal iqs

    Args:
        freq_ids (np.array(freq_nuum)): list of freq ids (0-40)
        dists (np.array(dist_num)): array of possible distances (m)
    """
    c = 299792458

    res = np.zeros((freqs.shape[0], dists.shape[0]))
    # omegas = 2.0*np.pi*np.arange(freq_0, freq_0+freq_ids.shape[0]*1e6, 1e6)
    # for freq_idx in range(freqs.shape[0]):
    #     omega = 2.0 * np.pi * freqs[freq_idx]
    #     # IQ = e^(-i*omega*t) = e^(-i*omega*l/c)
    #     res[freq_idx, :] = np.exp(-1j * dists * omega / c).real
    # return res

    return np.array([np.exp(-1j * dist * freqs * 2.0 * np.pi / C_SPEED).real for dist in dists]).T


DISTS = np.arange(0, 20, 0.01)

def estimate_dist(signals_data):
    dists = DISTS
    # plot_idxs = np.arange(0, signals_data.shape[1], 1)
    dist_probs = np.zeros((signals_data.shape[1], max(dists.shape)))
    # for t_idx in trange(0, max(measure_timestamps.shape)):
    # stear_vects = calc_stear_vect(FREQS, 2 * dists)
    stear_vects = np.array([np.exp(-1j * dist * FREQS * 2.0 * np.pi / C_SPEED).real for dist in dists]).T
    print(stear_vects.shape)
    for t_idx in range(
        0, signals_data.shape[1]
    ):
        iqs = np.expand_dims(signals_data[:, t_idx], axis=0)
        corr_matrix = iqs * iqs.conj().T

        corr_matrix += 1e-10 * np.identity(signals_data.shape[0])

        dist_corrs = my_correlation_use(stear_vects, corr_matrix)
        dist_probs[t_idx, :] = (dist_corrs - dist_corrs.min()) / (
            dist_corrs.max() - dist_corrs.min()
        )
    #  TODO: Why there are small lines? <10-01-22, astadnik> #
    return dist_probs

# def estimate_dist(signals_data):


SIGNAL_STD = 0.0

def simul_signals_shift_full(anchor, target)->np.ndarray:
    virt_targets = [
        (target, 1),
        (np.array([26 - target[0], target[1], target[2]]), 1),
        (np.array([13, target[1], target[2]]), 1)
        ]
    signal_v = np.zeros(FREQ_NUM, dtype=np.complex128)
    for virt_target, reflection_coeff in virt_targets:
        path_vector = virt_target - anchor
        path_len = np.sqrt(path_vector.dot(path_vector))

        # print(f"path_len:{path_len}", end="\n")

        omega_v = 2*np.pi*FREQS
        # print(omega_v.shape)
        a_v = C_SPEED / (2 * path_len * omega_v)  # amplitude from distance
        # a_v = 1  # amplitude from distance

        phi_v = (path_len * omega_v) / C_SPEED  # phase from distance
        signal_v += reflection_coeff * a_v * (np.exp(-1j * phi_v)+ np.random.normal(0, SIGNAL_STD)+1j*np.random.normal(0, SIGNAL_STD))
        #  + np.random.normal(0, SIGNAL_STD)+1j*np.random.normal(0, SIGNAL_STD))

    return signal_v

MEASURES_PER_S = 100

def main():
    x0 = np.array([5,0,0])
    tss = np.linspace(0, 9, MEASURES_PER_S*10) # 3 per s
    xss = np.array([x0 + t*np.array([.5,0,0]) for t in tss])
    print(xss[-1])
    # xss = x0 - 0.5*tss

    # signals = simul_signals_shift_full(np.array([0,0,0]), x)
    signals = np.array([simul_signals_shift_full(np.array([0,0,0]), x) for x in xss])

    dist_probs = np.array([
        fft(t_signals, FREQS)[0] for t_signals in signals
    ])
    print(signals.shape)
    dist_probs = estimate_dist(signals.T)
    print(dist_probs.shape)

    # fig, ax = plt.subplots()

    # dist_probs = dist_probs**2
    # plt.imshow(dist_probs.T, aspect="auto", cmap="inferno")
    # plt.imshow(dist_probs.T[::-1,:], aspect="auto", cmap="magma", extent = [5, 10, 0, 20])
    plt.imshow(dist_probs.T[::-1,:], aspect="auto", cmap="inferno", extent = [5, 10, 0, 20])
    plt.yticks([0, 5, 10, 15, 20])

    # plt.imshow(dist_probs.T[::-1,:], aspect="auto", cmap="plasma", extent = [5, 10, 0, 20])
    plt.xlabel('Distance, m')
    plt.ylabel('Estimated distance, m')
    plt.colorbar()

    # fft_amps, fft_freqs = fft(signals[0,:], FREQS)
    fig, ax = plt.subplots()
    ax.plot(DISTS, dist_probs[MEASURES_PER_S*5])
    ax.get_yaxis().set_ticks([])

    plot_format_axes(ax)
    plt.ylabel('')
    
    ax.xaxis.set_ticks([0, 7.2, 13, 18.8])
    # ticks = {0:'0', 1:'$x_0/c$', 2:'$x_1/c$', 5:'$x_2/c$'}
    # ticks = {0:'0', 7.2:'$f_i V_0/c$', 13:'$f_iV_1/c$', 18.8:'$f_iV_2/c$'}
    ticks = {0:'0', 7.2:'$x_0/c$', 13:'$x_1/c$', 18.8:'$x_2/c$'}

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos: ticks[x] if x in ticks else ''))
    plt.xlabel('Estimated distance, m')
    # ax.plot(DISTS, dist_probs[MEASURES_PER_S*0])


    # ax.plot(fft_freqs[(fft_freqs>=0)*(6>=fft_freqs)], fft_amps[(fft_freqs>=0)*(6>=fft_freqs)], "-")


    plt.show()

    


if(__name__ == "__main__"):
    main()