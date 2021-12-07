import numpy as np
import numpy.matlib

from Utilities import generate_point, signals_model

def ctranspose(arr: np.ndarray) -> np.ndarray:
    # Explanation of the math involved:
    # x      == Real(X) + j*Imag(X)
    # conj_x == Real(X) - j*Imag(X)
    # conj_x == Real(X) + j*Imag(X) - 2j*Imag(X) == x - 2j*Imag(X)
    tmp = arr.transpose()
    return tmp - 2j*tmp.imag

def main():
    np.random.seed(10)
    key_veloc_kmh = 5
    # track = 10:-0.1:0.1
    start_point_m = 10
    # scenario_matrix = [0.1, 0, 0, 1]
    scenario_matrix = [0.2, 0, 0, 0.6]#LOS, floor, ceiling, wall
    # scenario_matrix = [0.2, 0, 0, 0]
    # scenario_matrix = [0.2, 0.3, 0.4, 0.6]
    scenario_noise = 0 # not used yet

    freq_numb = 40 # 2MHz step

    # freq_list = 0:2:freq_numb*2
    freq_list = np.arange(0, freq_numb*2, 2.0)
    print(freq_list)
    delta_t = 300e-6 # time to do one frequency IQ measurement

    # onepack_times = delta_t * 0:7 # times of measurement 8 frequencies in a pack
    onepack_times = np.array(delta_t * np.arange(0.0,7.0)) # times of measurement 8 frequencies in a pack
    print(onepack_times)

    # pack_times = 0:2.4e-3:7 # times of packs measurements staring
    pack_times = np.arange(0, 7, 2.4e-3) # times of packs measurements staring
    print(pack_times)

    # time_flow = repmat(onepack_times, length(pack_times), 1) + repmat(pack_times', 1, length(onepack_times));
    time_flow = np.matlib.repmat(onepack_times, len(pack_times), 1) + np.matlib.repmat(pack_times.transpose(), 1, len(onepack_times))

    # time_flow = time_flow';
    time_flow = ctranspose(time_flow)
    # time_flow = time_flow(:);
    time_flow = np.array(*time_flow)
    # # time_flow = 0:delta_t:delta_t*1000;

    spectr_algo = 1; # 1 - FFT, 2 - MUSIC

    nsig = 3; # MUSIC ORDER
    sm_order = 1; # R matrix smooth level
    # sampl_dist = 0:0.02:20; # Tested distances
    sampl_dist = np.array(np.arange(0, 20, 0.02)) # Tested distances

    freq_set_type = 1; # 1 - regular frequency step, 0 - random frequency
    f_pack_len = 5; # for regular frequency step
    # freq_meas_set = 0:10:70; # for regular frequency step
    freq_meas_set = np.array(np.arange(0,70,10)); # for regular frequency step

    ## Define noise and delay errors
    delays=[]; # In this case default numbers will be used
    noises=[]; # In this case default numbers will be used

    ###############################
    # spec_1_evol = zeros(length(time_flow), length(sampl_dist));
    spec_1_evol = np.zeros((len(time_flow), len(sampl_dist)))
    # freq_meas_coll = NaN(freq_numb, length(time_flow));
    freq_meas_coll = np.resize(np.array(np.nan), (freq_numb, len(time_flow))) 
    # dist = zeros(min(length(time_flow), length(time_flow)), 4);
    dist = np.zeros(min(len(time_flow), len(time_flow)), 4)

    ## Determine signals along track
    # for t_idx = 1:length(time_flow)
    for t_idx in range(1, len(time_flow)):
        # if freq_set_type == 1
        if freq_set_type == 1:
        #     f1_idx = mod(t_idx - 1, length(freq_meas_set));
            f1_idx = np.mod(t_idx - 1, len(freq_meas_set))
        #     f2_idx = mod(floor((t_idx - 1) / length(freq_meas_set)), f_pack_len);
            f2_idx = np.floor((t_idx - 1) / len(freq_meas_set))%f_pack_len
        #     curr_freq = freq_meas_set(f1_idx + 1) + f2_idx * 2;
            curr_freq = float(freq_meas_set[f1_idx]) + f2_idx * 2
        # else
        else:
        #     curr_freq = (randi(freq_numb) - 1) * 2;
            curr_freq = float(np.random.uniform(0, freq_numb) - 1) * 2

        # end

        # [dist(t_idx, :), ampl_coeff] = generate_point(start_point_m, time_flow(t_idx), key_veloc_kmh, scenario_matrix, scenario_noise);
        dist[t_idx], ampl_coeff = generate_point(start_point_m, time_flow[t_idx], key_veloc_kmh, scenario_matrix, scenario_noise)
        # [~, signals] = signals_model(curr_freq, dist(t_idx, :), ampl_coeff, delays, noises);
        _, signals = signals_model(curr_freq, dist[t_idx, :], ampl_coeff, delays, noises)
        # freq_meas_coll(:, t_idx) = NaN;
        freq_meas_coll[:, t_idx] = np.Nan
        # freq_meas_coll(round(curr_freq / 2) + 1, t_idx) = signals;
        freq_meas_coll[round(curr_freq / 2), t_idx] = signals
    # end

    ## Estimate distances
    # iq_vect = NaN(freq_numb, 1);
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
        
    #     if sm_order > 1
    #         r_fin = spsmooth(r, sm_order);
    #     else
    #         r_fin = r;
    #     end
        
    #     sv = calc_stear_vect(freq_list, 2 * sampl_dist);
        
    #     r_size = size(r_fin, 1);
    #     switch spectr_algo
    #         case 1
    #             [spec, spec_1] = my_correlation_use(sv(1:r_size, :), iq_vect, r_fin);
    #         case 2
    #             [eigenvects,eigenvals,~, ~] = MYmusicdoa_eigen_det(r_fin,nsig,'ScanAngles',80*sampl_dist/max(sampl_dist)); # 'ScanAngles' - just to skip verifications
    #             [spec, spec_1] = MYmusicdoa_eigen_use(nsig, sv(1:r_size, :), eigenvects);
    #         otherwise
    #             disp('No such algorithm');
    #             return
    #     end
        
    #     spec_1_evol(floor((t_idx-1)/length(onepack_times) + 1), :) = (spec - min(spec)) / (max(spec) - min(spec));
    # end

    # [X, Y] = meshgrid(sampl_dist, dist(1:length(onepack_times):end, 1));
    # # spec_1_av_evol(spec_1_av_evol < 1e-5) = 1e-5;
    # # spec_1_evol = log(spec_1_evol);
    # figure, surf(X, Y, spec_1_evol(1:floor((t_idx-1)/length(onepack_times) + 1), :), 'LineStyle', 'none')
    # xlabel('Estimated distance, m')
    # ylabel('Real distance, m')
    # a = -28.2446;
    # b = 72.6000;
    # view(a, b);






main()