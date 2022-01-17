import numpy as np

from distance_determination import estimate_dist, simulate_signals
from parameters import Parameters
from utilities.data import dump_experiment

np.random.seed(10)


def experiment1():
    params = Parameters()
    params.freq_set_type = 1
    return params


def main():
    params = experiment1()
    dist, signals_data = simulate_signals(params)
    dist_probs = estimate_dist(signals_data, params)
    dump_experiment("default", params, dist, signals_data, dist_probs)


if __name__ == "__main__":
    main()
