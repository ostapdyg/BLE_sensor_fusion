import numpy as np
import plotly.express as px


def vis_dist_probs(dist_probs: np.ndarray, dist):
    dist_ideal = np.arange(0, 20, 0.02)
    return px.imshow(dist_probs.T, aspect="auto", y=dist_ideal, x=dist[::8, 0])
