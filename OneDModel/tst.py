from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

import matplotlib.pyplot as plt

PARTICLE_N = 100
# particles = pf_generate_uniform(
    # x_range=(0, 10), vel_range=(0, 3), rot_range=(0, 0.2), N=PARTICLE_N)
xs = np.arange(0, 10, 0.1)
weights = np.ones(PARTICLE_N) / PARTICLE_N

# probs = scipy.stats.norm(5000, 1000).pdf(np.arange(0, 10000))
# print(probs)
# # fig, ax = plot_pdf(probs)

signal_vals = simul_signals_shift_full(np.array([0, 0]), np.array([0, 5]))
#  + simul_signals_shift_full(np.array([0, 0]), np.array([0, 10]))*2
prob_f = estimate_dist_probf(signal_vals)
dists = np.arange(-10, 10, 0.01)

fig, ax = plt.subplots()
# ax.plot(dists, prob_f(dists))
# ax.plot(dists, prob_f(dists)**2)
# probs =  (lambda d:(prob_f(d)))(dists)
# ax.plot(dists, probs/np.sum(probs))
probs =  (lambda d:(prob_f(d)**2))(dists)
ax.plot(dists, probs/np.sum(probs))

ax.grid(True)
# pf_update_distr(weights, xs, probs)
# print(weights)
# fig, ax = plot_pdf(weights)

plt.show()