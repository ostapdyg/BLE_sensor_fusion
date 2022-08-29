from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

import matplotlib.pyplot as plt
import scipy.stats

BASE_FREQ = 2.4e9 #Hz
FREQ_STEP = 2e6 #Hz
FREQ_NUM = 80

FREQS = np.arange(BASE_FREQ, BASE_FREQ+FREQ_STEP*FREQ_NUM-1, FREQ_STEP)
C_SPEED = 299_792_458

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


# signal_vals = simul_signals_shift_full(np.array([0, 0]), np.array([0, 5]))

# prob_f = estimate_dist_probf(signal_vals)
# dists = np.arange(-10, 10, 0.01)

# probs = fft_f(particle_dist)
# pf_update_distr(weights, probs)
# 

# ble_dist = estimate_distance(signal_vals)
# prob_f = lambda x:scipy.stats.norm(0, 0.1).pdf(ble_dist - x)
# signal_vals = simul_signals_shift_full(np.array([0, 0]), np.array([0, 2]))
signal_vals = simul_signals_shift_full(np.array([1e-100, 1e-100]), np.array([0, 2])) + \
        simul_signals_shift_full(np.array([1e-100, 1e-100]), np.array([0, 4]))
prob_xss = np.arange(-10, 10, 0.01)
prob_f = estimate_dist_probf(signal_vals)
# def estimate_dist_probf(signal_v, freqs=FREQS):
#     signal_interp = signal_v
#     fft_vals, fft_freqs = fft(signal_interp, freqs)
#     # dists = np.arange(0, dist_max*100)

fig, ax = plt.subplots()
ax.grid(True)

ax.plot(prob_xss, prob_f(prob_xss))


fig, ax = plt.subplots()
ax.grid(True)

fft_vals, fft_freqs = fft(signal_vals, FREQS)
ax.plot(fft_freqs*C_SPEED, fft_vals)



# plot_particles(np.array([xs]), np.array([weights]))

# probs =  (lambda d:(prob_f(d)**2))(dists)

# pf_update_distr(weights, prob)

# ax.scatter(xs, weights)
# print(weights)
# fig, ax = plot_pdf(weights)

plt.show()