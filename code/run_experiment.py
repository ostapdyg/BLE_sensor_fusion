import numpy as np

from distance_determination import estimate_dist, simulate_signals
from simul.parameters import Parameters
from simul.utilities.data import dump_experiment
from simul.vis.dist_probs import vis_dist_probs
from simul.vis.signals import vis_signals
from simul.vis.signals import vis_signals_2d

np.random.seed(10)


experiments = {
    "default": Parameters(freq_set_type=1),
    "default_full": Parameters(freq_set_type=2),
    "full_no_walls":Parameters(freq_set_type=2, scenario_matrix=[1.0, 0.0, 0.0, 0.0])
    # "default_full": Parameters(freq_set_type=2),

}


def main():
    # exp_name = "default_full"
    exp_name = "full_no_walls"
    params = experiments[exp_name]
    dist, signals_data = simulate_signals(params)
    vis_signals_2d(signals_data, dist, params, n=-1).show()
    # print(params.tss, signals_data)
    dist_probs = estimate_dist(signals_data, params)
    # fig_amp, fig_angle, fig_reals = vis_signals(signals_data, dist, params, dump=True)
    # fig_amp.show()
    # fig_angle.show()
    # vis_dist_probs(dist_probs, dist, params)

    # dump_experiment(exp_name, params, dist, signals_data, dist_probs)


if __name__ == "__main__":
    main()
