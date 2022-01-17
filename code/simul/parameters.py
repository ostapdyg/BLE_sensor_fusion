from dataclasses import dataclass

import numpy as np


def calc_measure_timestamps(delta_t):
    # times of measurement 8 frequencies in a pack
    onepack_times = delta_t * np.arange(0.0, 8.0)
    # times of packs measurements staring
    pack_times = np.arange(0, 7, 2.4e-3).T[:, np.newaxis]
    row, col = len(pack_times), len(onepack_times)

    # pack_times.conj() == pack_times
    measure_ts = np.tile(onepack_times, (row, 1)) + np.tile(pack_times, (1, col))

    return measure_ts.flatten()


@dataclass
class Parameters:
    vel = 5
    start_pos = 10

    # scenario_matrix = [0.1, 0, 0, 1]
    scenario_matrix = [0.2, 0.0, 0.0, 0.6]  # LOS, floor, ceiling, wall
    # scenario_matrix = [0.2, 0, 0, 0]
    # scenario_matrix = [0.2, 0.3, 0.4, 0.6]
    scenario_noise = 0  # not used yet

    n_freq = 40  # 2MHz step

    freqs = np.arange(0, n_freq * 2, 2.0)

    # spectr_algo = 1; # 1 - FFT, 2 - MUSIC

    freq_set_type = 2
    # 1 - regular frequency step, 0 - random frequency, 2 - all simultaneously
    f_pack_len = 5
    # for regular frequency step
    freq_meas_set = np.arange(0, 80, 10)
    # for regular frequency step

    ## Define noise and delay errors
    # In this case default numbers will be used
    delays = np.array([])
    noises = np.array([])

    delta_t = 300e-6  # time to do one frequency IQ measurement
    tss = calc_measure_timestamps(delta_t)
