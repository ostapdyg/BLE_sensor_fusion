import numpy as np

from distance_determination import estimate_dist, simulate_signals
from simul.parameters import Parameters
from simul.utilities.data import dump_experiment
from simul.vis.dist_probs import vis_dist_probs
from simul.vis.signals import vis_signals

# from simul.vis.signals import vis_signals_2d
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(10)


experiments = {
    "default": Parameters(freq_set_type=1),
    "default_full": Parameters(freq_set_type=2),
    "default_random": Parameters(freq_set_type=0),

    # "default_noice"
    #     "default_noice_full"
    "full_no_walls": Parameters(freq_set_type=2, scenario_matrix=[1.0, 0.0, 0.0, 0.0]),
    "no_walls": Parameters(freq_set_type=1, scenario_matrix=[1.0, 0.0, 0.0, 0.0], ),
    "no_walls_random": Parameters(freq_set_type=0, scenario_matrix=[1.0, 0.0, 0.0, 0.0]),

    # "default_full": Parameters(freq_set_type=2),
}


def vis_signals_3d(
    signals_data: np.ndarray,
    dist: np.ndarray,
    p: Parameters,
    n_freq=40,
    n=400,
    dump=False,
    dump_dir="../graphs"
):
    # TODO: Fix for Windows?

    #  TODO: Plot different graph if the data contains nans <26-01-22, astadnik> #
    # assert np.all(~np.isnan(signals_data))
    amp, angle = np.abs(signals_data[:n_freq, :n]), np.angle(signals_data[:n_freq, :n])
    reals = np.real(signals_data[:n_freq, :n])
    y = p.freqs[:n_freq]
    x = dist[:n, 0]
    print(amp.shape, angle.shape, reals.shape, x.shape, y.shape)
    fig_amp = go.Figure(data=[go.Surface(z=amp, x=x, y=y)]).update_layout(
        scene=dict(xaxis_title="Distance", yaxis_title="Freq", zaxis_title="Signal amp")
    )
   
    fig_angle = go.Figure(data=[go.Surface(z=angle, x=x, y=y)]).update_layout(
        scene=dict(
            xaxis_title="Distance", yaxis_title="Freq", zaxis_title="Signal angle"
        )
    )
    

    fig_reals = go.Figure(data=[go.Surface(z=reals, x=x, y=y/2,    
        contours = {
            # "x": {"show": True, "start": 10, "end": 9.98, "size": 0.04,},
            # "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        },
    )]).update_layout(
        scene=dict(
            xaxis_title="Position/Time", yaxis_title="Channel", zaxis_title="Signal real part", zaxis=dict(showticklabels=False)
        ),

    ).update_layout(
        scene = {
            "xaxis": {"nticks": 10},
            "zaxis": {"nticks": 4},
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        }).update_traces(contours_x=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_x=True),
                        contours_y=dict(show=True, usecolormap=True,
                                 highlightcolor="limegreen", project_y=True))

    return fig_amp, fig_angle, fig_reals


def main():
    # exp_name = "default_full"
    exp_name = "full_no_walls"
    params = experiments[exp_name]

    dist, signals_data = simulate_signals(params)
    # vis_signals(params.tss, signals_data, dist).show()
    # print(params.tss, signals_data)
    # dist_probs = estimate_dist(signals_data, params)
    # fig_amp, fig_angle, fig_reals = vis_signals(signals_data, dist, params, dump=True)
    # fig_amp.show()
    # fig_angle.show()
    # vis_dist_probs(dist_probs, dist, params).show()
    fig_amp, fig_angle, fig_reals = vis_signals_3d(signals_data, 10-dist, params, dump=True)
    fig_reals.show()
    # dump_experiment(exp_name, params, dist, signals_data, dist_probs)


if __name__ == "__main__":
    main()
