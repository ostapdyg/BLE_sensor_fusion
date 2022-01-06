import logging

import numpy as np
import plotly.express as px
from icecream import ic
from tqdm.auto import trange

from Utilities.calc_stear_vect import calc_stear_vect
from Utilities.generate_point import generate_point
from Utilities.my_correlation_use import my_correlation_use
from Utilities.parameters import Parameters
from Utilities.signals_model import signals_model

logger = logging.getLogger(__name__)


def simulate_signals(p: Parameters):
    freq_meas_coll = np.full(
        (p.freq_numb, len(p.measure_timestamps)), np.NaN, np.csingle
    )

    # dist = zeros(min(length(measure_timestamps), length(measure_timestamps)), 4);
    dist = np.zeros((len(p.measure_timestamps), 4))

    for t_idx in trange(
        0,
        len(p.measure_timestamps),
        desc="Simulating distances and signals",
        leave=False,
    ):
        #  TODO: Write comments about this, I again don't understand how it
        #  works <06-01-22, astadnik> #
        if p.freq_set_type == 1:
            f1_idx = t_idx % len(p.freq_meas_set)
            f2_idx = np.floor((t_idx) / len(p.freq_meas_set)) % p.f_pack_len
            curr_freq = p.freq_meas_set[f1_idx] + f2_idx * 2
        else:
            curr_freq = float(np.random.uniform(0, p.freq_numb) - 1) * 2

        dist[t_idx, :], ampl_coeff = generate_point(
            p.start_point_m,
            p.measure_timestamps[t_idx],
            p.key_veloc_kmh,
            p.scenario_matrix,
            p.scenario_noise,
        )
        _, signals = signals_model(
            curr_freq, dist[t_idx : t_idx + 1, :], ampl_coeff, p.delays, p.noises
        )
        freq_meas_coll[:, t_idx] = np.NaN
        freq_meas_coll[round(curr_freq / 2), t_idx] = signals[0]
    return dist, freq_meas_coll


def main():
    np.random.seed(10)
    params = Parameters()

    ###############################
    # spec_1_evol = zeros(length(measure_timestamps), length(sampl_dist));
    # spec_1_evol = np.zeros((len(measure_timestamps), len(sampl_dist)))
    # freq_meas_coll = NaN(freq_numb, length(measure_timestamps));
    dist, freq_meas_coll = simulate_signals(params)

    # data_1 = freq_meas_coll[1, ~np.isnan(freq_meas_coll[1, :])]
    # amplitudes = np.abs(data_1)
    # xs = (key_veloc_kmh / 3600) * 1000 * measure_timestamps[~np.isnan(freq_meas_coll[1, :])]
    # angles = np.angle(data_1)
    # px.scatter(y = amplitudes, x= xs).show()
    # px.scatter(y = angles, x= xs).show()

    # Interpolate NaNs
    iq_data = np.array(freq_meas_coll, copy=True)
    for freq_i in trange(iq_data.shape[0]):
        last_val = 0 + 0j
        for t_i in range(iq_data.shape[1]):
            if np.isnan(iq_data[freq_i, t_i]):
                iq_data[freq_i, t_i] = last_val
            else:
                last_val = iq_data[freq_i, t_i]
    # px.scatter(y = np.real(iq_data[1,:]), x= range(measure_timestamps.shape[0])).show()
    # px.scatter(y = np.imag(iq_data[1,:]), x= range(measure_timestamps.shape[0])).show()

    # Estimate distances
    dists = np.arange(0, 20, 0.02)
    plot_idxs = np.arange(0, max(params.measure_timestamps.shape), 8)
    dist_probs = np.zeros((max(plot_idxs.shape), max(dists.shape)))
    # for t_idx in trange(0, max(measure_timestamps.shape)):
    for t_idx in trange(0, max(plot_idxs.shape)):
        iqs = np.array([iq_data[:, plot_idxs[t_idx]]])
        corr_matrix = iqs * iqs.conj().T
        # print(f"corr_matrix:{corr_matrix.shape}")
        corr_matrix += 1e-10 * np.identity(iq_data.shape[0])
        # VerifyCorrMatrix(corr_matrix)

        stear_vects = calc_stear_vect(params.freq_list, 2 * dists)
        dist_corrs = my_correlation_use(stear_vects, corr_matrix)
        dist_probs[t_idx, :] = (dist_corrs - dist_corrs.min()) / (
            dist_corrs.max() - dist_corrs.min()
        )
    #  TODO: Plot the ground truth(dist) as well <06-01-22, astadnik> #
    px.imshow(dist_probs, aspect="auto").show()
    # px.
    # print(dist_probs)
    #     sv = calc_stear_vect(freq_list, 2 * sampl_dist);

    # for t_idx = 1:length(onepack_times):length(measure_timestamps)
    #     disp(100 * t_idx / length(measure_timestamps))

    #     for tm_idx = 0:length(onepack_times)-1
    #         sign_idx = find(~isnan(freq_meas_coll(:, t_idx + tm_idx)));
    #         iq_vect(sign_idx) = freq_meas_coll(sign_idx, t_idx + tm_idx);
    #     end
    #     if sum(isnan(iq_vect)) == 0
    #         r_t = iq_vect * iq_vect';
    #     else
    #         continue;
    #     end
    #     r_noise = diag(1e-10 * ones(size(iq_vect)));
    #     r = r_t + r_noise; # required only in noiseless cases to make shure the corellation matrix is posively defined
    #     VerifyCorrMatrix(r); # verify if the correlation r matrix is OK

    #     sv = calc_stear_vect(freq_list, 2 * sampl_dist);

    #     r_size = size(r_fin, 1);
    #     switch spectr_algo
    #     [spec, spec_1] = my_correlation_use(sv(1:r_size, :), iq_vect, r_fin);
    #     end
    #     spec_1_evol(floor((t_idx-1)/length(onepack_times) + 1), :) = (spec - min(spec)) / (max(spec) - min(spec));
    # end


main()
