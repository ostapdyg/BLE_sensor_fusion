from dataclasses import dataclass, field

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


def parallel_wall_dist(x:float, t:float, wall_pos:float)->float:
    return 2 * ((x / 2) ** 2 + (wall_pos) ** 2) ** 0.5

def normal_wall_dist(x:float, t:float, wall_pos:float)->float:
    # return 2 * abs(wall_pos - x) + x
    # X -> wall -> receiver
    return abs(wall_pos - x) + abs(wall_pos)


@dataclass(kw_only=True)
class Parameters:
    vel: int = 5
    start_pos: int = 10
    key_x0 = 10

    # scenario_matrix = [0.1, 0, 0, 1]
    scenario_matrix: list[float] = field(
        default_factory=lambda: [0.2, 0.0, 0.0, 0.6]
    )  # LOS, floor, ceiling, wall
    # scenario_matrix = [0.2, 0, 0, 0]
    # scenario_matrix = [0.2, 0.3, 0.4, 0.6]

    dist_funcs:list[function] = [
        lambda x,t:abs(x),                      # LOS
        lambda x,t:parallel_wall_dist(x, t, 1.0), # floor
        lambda x,t:parallel_wall_dist(x, t, 4.0), # ceiling
        lambda x,t:normal_wall_dist(x, t, 13.0),  # wall
    ]

    scenario_noise = 0  # not used yet

    n_freq: int = 40  # 2MHz step

    freqs: np.ndarray = field(default=np.arange(0, n_freq * 2, 2.0), repr=False)

    # spectr_algo = 1; # 1 - FFT, 2 - MUSIC

    # freq_set_type = 1
    freq_set_type = 2
    # 0 - random frequency,
    # 1 - regular frequency step,
    # 2 - all frequencies at each timestamp
    f_pack_len = 5

    # freq_meas_set = np.arange(0, 80, 10)  #Not used

    ## Define noise and delay errors
    # In this case default numbers will be used
    delays: np.ndarray = np.array([])
    noises: np.ndarray = np.array([])

    delta_t: float = 300e-6  # time to do one frequency IQ measurement
    tss: np.ndarray = field(default=calc_measure_timestamps(delta_t), repr=False)
