# import logging

import numpy as np
from tqdm.auto import tqdm, trange

from simul.distances.calc_stear_vect import calc_stear_vect
from simul.distances.my_correlation_use import my_correlation_use
from simul.parameters import Parameters
from simul.signals.generate_point import generate_point
from simul.signals.signals_model import signals_model
from simul.utilities.data import dump_experiment
from simul.vis.dist_probs import vis_dist_probs
from simul.vis.signals import vis_signals, vis_fft

# logger = logging.getLogger(__name__)


def get_current_freq(ts_i, p: Parameters) -> np.ndarray | float:
    """
    big steps by 10, small steps by 2

    :param ts_i [TODO:type]: [TODO:description]
    :param p Parameters: [TODO:description]
    """
    if p.freq_set_type == 1:
        f2_i = np.floor(ts_i / 8) % p.f_pack_len
        return (ts_i % 8) * 10 + f2_i * 2
    if p.freq_set_type == 2:
        return np.arange(0, p.n_freq) * 2
    return float(np.random.uniform(0, p.n_freq) - 1) * 2


def simulate_signals(p: Parameters):
    signals_data = np.full((p.freqs.size, len(p.tss)), np.NaN, dtype=np.csingle)

    # Distance LOS, floor, ceiling, wall
    dist = np.zeros((len(p.tss), 4))

    desc = "Simulating distances and signals"
    for ts_i, ts in enumerate(tqdm(p.tss, desc=desc, leave=False)):
        curr_freq = get_current_freq(ts_i, p)

        dist[ts_i, :], ampl_coeff = generate_point(p, ts)
        _, signals = signals_model(curr_freq, dist[ts_i : ts_i + 1, :], ampl_coeff, p)
        signals_data[:, ts_i] = np.NaN
        if isinstance(curr_freq, np.ndarray):
            signals_data[curr_freq // 2, ts_i] = signals[0]
        else:
            signals_data[int(curr_freq // 2), ts_i] = signals[0]
    return dist, signals_data


def interpolate_NAN(signals_data):
    iq_data = signals_data.copy()
    for freq_i in trange(iq_data.shape[0], desc="Interpolating NaNs", leave=False):
        last_val = 0 + 0j
        for t_i in range(iq_data.shape[1]):
            if np.isnan(iq_data[freq_i, t_i]):
                iq_data[freq_i, t_i] = last_val
            else:
                last_val = iq_data[freq_i, t_i]
    return iq_data


def estimate_dist(signals_data, params: Parameters):
    iq_data = interpolate_NAN(signals_data)
    # Estimate distances
    dists = np.arange(0, 20, 0.02)
    # Indexes of timestamps
    plot_idxs = np.arange(0, len(params.tss), 8)
    dist_probs = np.zeros((max(plot_idxs.shape), max(dists.shape)))
    # for t_idx in trange(0, max(measure_timestamps.shape)):
    stear_vects = calc_stear_vect(params.freqs, 2 * dists)
    for t_idx in trange(
        0, max(plot_idxs.shape), desc="Searching for distances", leave=False
    ):
        iqs = np.expand_dims(iq_data[:, plot_idxs[t_idx]], axis=0)
        corr_matrix = iqs * iqs.conj().T

        corr_matrix += 1e-10 * np.identity(iq_data.shape[0])

        dist_corrs = my_correlation_use(stear_vects, corr_matrix)
        dist_probs[t_idx, :] = (dist_corrs - dist_corrs.min()) / (
            dist_corrs.max() - dist_corrs.min()
        )
    #  TODO: Why there are small lines? <10-01-22, astadnik> #
    return dist_probs


def main():
    np.random.seed(10)
    params = Parameters()
    params.freq_set_type = 2

    dist, signals_data = simulate_signals(params)
    dist_probs = estimate_dist(signals_data, params)

    dump_experiment("default", params, dist, signals_data, dist_probs)

    vis_dist_probs(dist_probs, dist, params)
    vis_fft(signals_data[0,:], params.tss).show()
    vis_fft(signals_data[:,0], params.freqs).show()


if __name__ == "__main__":
    main()
