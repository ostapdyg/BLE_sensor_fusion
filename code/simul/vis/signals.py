# import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from simul.parameters import Parameters


def vis_signals(
    signals_data: np.ndarray, dist: np.ndarray, p: Parameters, n=200, dump=False
):
    amp, angle = np.abs(signals_data[:, :n]), np.angle(signals_data[:, :n])
    y = p.freqs
    x = dist[:n, 0]
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
    return fig_amp, fig_angle
