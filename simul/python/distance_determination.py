import numpy as np
import plotly.express as px
import plotly.io as pio
from tqdm.auto import trange

pio.renderers.default = "browser"
import numpy.matlib

from Utilities import *


def main():
    np.random.seed(10)
    key_veloc_kmh = 5
    start_point_m = 10

    # scenario_matrix = [0.1, 0, 0, 1]
    scenario_matrix = [0.2, 0.0, 0.0, 0.6]  # LOS, floor, ceiling, wall
    # scenario_matrix = [0.2, 0, 0, 0]
    # scenario_matrix = [0.2, 0.3, 0.4, 0.6]
    scenario_noise = 0  # not used yet

    freq_numb = 40  # 2MHz step

    freq_list = np.arange(0, freq_numb * 2, 2.0)
    delta_t = 300e-6  # time to do one frequency IQ measurement
    # times of measurement 8 frequencies in a pack
    onepack_times = delta_t * np.arange(0.0, 8.0)
    pack_times = np.arange(0, 7, 2.4e-3)  # times of packs measurements staring
    print(f"{pack_times.shape=}")
    print(f"{onepack_times.shape=}")
    # time_flow = repmat(onepack_times, length(pack_times), 1) + repmat(pack_times', 1, length(onepack_times));
    # print(ctranspose(pack_times) == pack_times.conj().T)

    time_flow = np.tile(onepack_times, (len(pack_times), 1)) + np.tile(
        pack_times.conj().T[:, None], (1, len(onepack_times))
    )
    print(time_flow.shape)
    print(f"time_flow:{time_flow.shape}{time_flow[0,:]}")
    print(f"time_flow:{time_flow.shape}{time_flow[0,:]}")

    # time_flow = time_flow';
    time_flow = time_flow.transpose()
    # time_flow = time_flow(:);
    time_flow = time_flow.flatten("F")
    # # time_flow = 0:delta_t:delta_t*1000;
    print(f"time_flow:{time_flow.shape}{time_flow[0:8]}")
    # spectr_algo = 1; # 1 - FFT, 2 - MUSIC

    # nsig = 3; # MUSIC ORDER
    # sm_order = 1; # R matrix smooth level
    # sampl_dist = 0:0.02:20; # Tested distances
    # sampl_dist = np.array(np.arange(0, 20, 0.02)) # Tested distances

    freq_set_type = 1
    # 1 - regular frequency step, 0 - random frequency
    f_pack_len = 5
    # for regular frequency step
    # freq_meas_set = 0:10:70; # for regular frequency step
    freq_meas_set = np.array(np.arange(0, 80, 10))
    # for regular frequency step

    ## Define noise and delay errors
    delays = np.array([])
    # In this case default numbers will be used
    noises = np.array([])
    # In this case default numbers will be used

    ###############################
    # spec_1_evol = zeros(length(time_flow), length(sampl_dist));
    # spec_1_evol = np.zeros((len(time_flow), len(sampl_dist)))
    # freq_meas_coll = NaN(freq_numb, length(time_flow));
    freq_meas_coll = np.resize(
        np.array(np.NaN, dtype=np.csingle), (freq_numb, len(time_flow))
    )
    print(f"freq_meas_coll:{freq_meas_coll.shape}")

    # dist = zeros(min(length(time_flow), length(time_flow)), 4);
    dist = np.zeros((max(time_flow.shape), 4))
    print(f"dist:{dist.shape}")
    # print(f"    gen_p dist:{dist.shape}")

    ## Determine signals along track
    # for t_idx = 1:length(time_flow)
    print(f"    freq_meas_set:{freq_meas_set}")

    for t_idx in trange(0, max(time_flow.shape)):
    # for t_idx in range(0, 500):

        # for t_idx in range(0,1):

        # if freq_set_type == 1
        if freq_set_type == 1:
            #     f1_idx = mod(t_idx - 1, length(freq_meas_set));
            f1_idx = np.mod(t_idx, len(freq_meas_set))
            #     f2_idx = mod(floor((t_idx - 1) / length(freq_meas_set)), f_pack_len);
            f2_idx = np.floor((t_idx) / len(freq_meas_set)) % f_pack_len
            #     curr_freq = freq_meas_set(f1_idx + 1) + f2_idx * 2;
            curr_freq = freq_meas_set[f1_idx] + f2_idx * 2
        # else
        else:
            #     curr_freq = (randi(freq_numb) - 1) * 2;
            curr_freq = float(np.random.uniform(0, freq_numb) - 1) * 2

        # end

        # [dist(t_idx, :), ampl_coeff] = generate_point(start_point_m, time_flow(t_idx), key_veloc_kmh, scenario_matrix, scenario_noise);
        dist[t_idx, :], ampl_coeff = generate_point(
            start_point_m,
            time_flow[t_idx],
            key_veloc_kmh,
            scenario_matrix,
            scenario_noise,
        )
        # [~, signals] = signals_model(curr_freq, dist(t_idx, :), ampl_coeff, delays, noises);
        _, signals = signals_model(
            curr_freq, dist[t_idx : t_idx + 1, :], ampl_coeff, delays, noises
        )
        # if(not PRINT_DIRT):print(f"    signals:{signals.shape}")

        # freq_meas_coll(:, t_idx) = NaN;
        freq_meas_coll[:, t_idx] = np.NaN
        # if(not PRINT_DIRT):print(f"    freq_meas_coll:{freq_meas_coll.shape}")

        # freq_meas_coll(round(curr_freq / 2) + 1, t_idx) = signals;
        freq_meas_coll[round(curr_freq / 2), t_idx] = signals[0]
        # print(f"        dist:{dist[t_idx, :]}\n    cur_freq:{curr_freq}, signals:{signals[0]}")
    # end

    # data_1 = freq_meas_coll[1, ~np.isnan(freq_meas_coll[1, :])]
    # amplitudes = np.abs(data_1)
    # xs = (key_veloc_kmh / 3600) * 1000 * time_flow[~np.isnan(freq_meas_coll[1, :])]
    # angles = np.angle(data_1)
    # px.scatter(y = amplitudes, x= xs).show()
    # px.scatter(y = angles, x= xs).show()

    # Interpolate NaNs
    iq_data = np.array(freq_meas_coll, copy=True)
    for freq_i in trange(iq_data.shape[0]):
        last_val = 0+0j
        for t_i in range(iq_data.shape[1]):
            if(np.isnan(iq_data[freq_i, t_i])):
                iq_data[freq_i, t_i] = last_val
            else:
                last_val = iq_data[freq_i, t_i]
    # px.scatter(y = np.real(iq_data[1,:]), x= range(time_flow.shape[0])).show()
    # px.scatter(y = np.imag(iq_data[1,:]), x= range(time_flow.shape[0])).show()

    # Estimate distances
    dists = np.arange(0, 20, 0.02)
    plot_idxs = np.arange(0, max(time_flow.shape),  max(onepack_times.shape))
    dist_probs = np.zeros((max(plot_idxs.shape), max(dists.shape)))
    # for t_idx in trange(0, max(time_flow.shape)):
    for t_idx in trange(0, max(plot_idxs.shape)):
        iqs = np.array([iq_data[:, plot_idxs[t_idx]]])
        corr_matrix = iqs*iqs.conj().T
        # print(f"corr_matrix:{corr_matrix.shape}")
        corr_matrix += 1e-10*np.identity(iq_data.shape[0])
        # VerifyCorrMatrix(corr_matrix)

        stear_vects = calc_stear_vect(freq_list, 2*dists)
        dist_corrs = my_correlation_use(stear_vects, corr_matrix)
        dist_probs[t_idx, :] = (dist_corrs - dist_corrs.min())/(dist_corrs.max() - dist_corrs.min())
    px.imshow(dist_probs, aspect='auto').show()
    # px.
    # print(dist_probs)
    #     sv = calc_stear_vect(freq_list, 2 * sampl_dist);


    # for t_idx = 1:length(onepack_times):length(time_flow)
    #     disp(100 * t_idx / length(time_flow))

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
