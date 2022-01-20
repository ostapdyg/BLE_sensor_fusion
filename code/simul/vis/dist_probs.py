import plotly.express as px

import numpy as np


def vis_dist_probs(dist_probs: np.ndarray, dist, params):
    dist_ideal = np.arange(0, 20, 0.02)
    px.imshow(dist_probs.T, aspect="auto", y = dist_ideal, x = dist[::8, 0]).show()
