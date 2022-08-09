from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

import matplotlib.pyplot as plt
import scipy.stats


def plot_particles(xs, weights):
    fig, ax = plt.subplots()
    ax.grid(True)
    dists = np.arange(0, 10, 0.1)
    for i in range(xs.shape[0]):
        kde = scipy.stats.gaussian_kde(xs[i,:], weights=weights[i, :])
        ax.plot(dists, kde(dists))

PARTICLE_N = 100

xs = np.arange(0, 10, 0.1)
weights = np.ones(PARTICLE_N) / PARTICLE_N


signal_vals = simul_signals_shift_full(np.array([0, 0]), np.array([0, 5]))

prob_f = estimate_dist_probf(signal_vals)
dists = np.arange(-10, 10, 0.01)


fig, ax = plt.subplots()
ax.grid(True)
ax.scatter(xs, weights)

# ax.plot(dists, probs/np.sum(probs))

plot_particles(np.array([xs]), np.array([weights]))

# probs =  (lambda d:(prob_f(d)**2))(dists)

# pf_update_distr(weights, prob)

# ax.scatter(xs, weights)
# print(weights)
# fig, ax = plot_pdf(weights)

plt.show()