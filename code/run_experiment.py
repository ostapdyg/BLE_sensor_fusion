import numpy as np

from distance_determination import estimate_dist, simulate_signals
from simul.parameters import Parameters
from simul.utilities.data import dump_experiment

np.random.seed(10)


experiments = {
    "default": Parameters(freq_set_type=1),
    "default_full": Parameters(freq_set_type=0),
}


def main():
    exp_name, params = experiments["default"]
    dist, signals_data = simulate_signals(params)
    dist_probs = estimate_dist(signals_data, params)
    # dump_experiment(exp_name, params, dist, signals_data, dist_probs)


if __name__ == "__main__":
    main()
