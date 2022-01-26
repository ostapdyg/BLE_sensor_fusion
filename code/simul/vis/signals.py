# import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from simul.parameters import Parameters


def vis_signals(
    signals_data: np.ndarray,
    dist: np.ndarray,
    p: Parameters,
    n_freq=20,
    n=100,
    dump=False,
):
    #  TODO: Plot different graph if the data contains nans <26-01-22, astadnik> #
    assert np.all(~np.isnan(signals_data))
    amp, angle = np.abs(signals_data[:n_freq, :n]), np.angle(signals_data[:n_freq, :n])
    reals = np.real(signals_data[:n_freq, :n])
    y = p.freqs[:n_freq]
    x = dist[:n, 0]
    print(amp.shape, angle.shape, reals.shape, x.shape, y.shape)
    fig_amp = go.Figure(data=[go.Surface(z=amp, x=x, y=y)]).update_layout(
        scene=dict(xaxis_title="Distance", yaxis_title="Freq", zaxis_title="Signal amp")
    )
    if dump:
        fig_amp.write_html("/tmp/amplitudes.html")
    fig_angle = go.Figure(data=[go.Surface(z=angle, x=x, y=y)]).update_layout(
        scene=dict(
            xaxis_title="Distance", yaxis_title="Freq", zaxis_title="Signal angle"
        )
    )
    if dump:
        fig_angle.write_html("/tmp/angles.html")

    fig_reals = go.Figure(data=[go.Surface(z=reals, x=x, y=y)]).update_layout(
        scene=dict(
            xaxis_title="Distance", yaxis_title="Freq", zaxis_title="Signal real part"
        )
    )
    if dump:
        fig_reals.write_html("/tmp/reals.html")
    return fig_amp, fig_angle, fig_reals
