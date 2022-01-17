import numpy as np

from distance_determination import estimate_dist, simulate_signals
from simul.parameters import Parameters
from simul.utilities.data import dump_experiment

np.random.seed(10)


def experiment1():
    params = Parameters()
    params.freq_set_type = 1
    return "default", params


def experiment2():
    params = Parameters()
    params.freq_set_type = 2
    return "default_full", params


def main():
    exp_name, params = experiment2()
    dist, signals_data = simulate_signals(params)
    dist_probs = estimate_dist(signals_data, params)
    dump_experiment(exp_name, params, dist, signals_data, dist_probs)


if __name__ == "__main__":
    main()
