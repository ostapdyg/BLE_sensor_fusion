import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
import numpy as np


def vis_signals(signals_data: np.ndarray):
    amp, angle = np.abs(signals_data[:, :1000]), np.angle(signals_data[:, :1000])
    fig = go.Figure(data=[go.Surface(z=amp)]).update_layout(
        scene=dict(xaxis_title="Ts", yaxis_title="Freq", zaxis_title="Signal amp")
    )
    fig.write_html('/tmp/amplitudes.html')
    fig.show()
    fig = go.Figure(data=[go.Surface(z=angle)]).update_layout(
        scene=dict(xaxis_title="Ts", yaxis_title="Freq", zaxis_title="Signal angle")
    )
    fig.write_html('/tmp/angles.html')
    fig.show()
