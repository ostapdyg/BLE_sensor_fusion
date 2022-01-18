import plotly.express as px

import numpy as np


def vis_dist_probs(dist_probs: np.ndarray):
    px.imshow(dist_probs[:, ::-1].T, aspect="auto").show()
