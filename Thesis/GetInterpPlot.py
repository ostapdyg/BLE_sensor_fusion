import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline


from Src.simul import simul_get_pos, simul_signals_shift_full
from Src.simul import ANCHOR_COORDS, FREQS, C_SPEED
from Src.estimate_dist import fft, interp_lsq_spline

from Src.plot import plot_format_axes



from scipy import signal, interpolate

np.random.seed(1337)
signal_vals_ideal = simul_signals_shift_full(np.array([5.1, 0, 0]), [0,0,0])
signal_vals_0 = signal_vals_ideal + np.random.randn(*signal_vals_ideal.shape)*0.0002

fig, ax = plt.subplots()

ax.plot(
    range(40),
    signal_vals_ideal.real,
    "--",
    label="Ideal",
)


signal_vals_0[15:31] = 0
signal_vals_0[15:31] = np.nan

ax.plot(
    range(40),
    signal_vals_0.real,
    ":b.",
    label="Received",
)


signal_vals_interp = interp_lsq_spline(
    np.arange(40)[~np.isnan(signal_vals_0)],
    np.arange(40),
    signal_vals_0[~np.isnan(signal_vals_0)],
)


ax.plot(
    range(40),
    signal_vals_interp.real,
    "-g",
    label="Interpolated",
)

plot_format_axes(ax, y=False)

ax.axvline(
    15,
    color="orange",
    linestyle="dashdot",
    alpha=0.5
    )
ax.axvline(    
    30,
    color="orange",
    linestyle="dashdot",
    alpha=0.5
)
ax.axvspan(15, 30, alpha=0.2, color='orange')
ax.text(16, 0.004, "Unused frequencies band")

def update_xticks(x, pos):
    return f'{2402+2*int(x)}MHz'

ax.get_xaxis().set_ticks([0, 15, 30, 39])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(update_xticks))

plt.legend()
# plt.show()


fig, ax = plt.subplots()


freq_data_interp, fft_freqs = fft(signal_vals_interp, FREQS)
dists = -fft_freqs*C_SPEED
dist_amps_interp = np.abs(freq_data_interp)
plot_format_axes(ax, y=False)
ax.axes.get_yaxis().set_ticks([])
ax.plot(dists, dist_amps_interp, label="Interpolated")

for x in (5.10 ,10.4, 14.4):
    ax.axvline(
        x,
        linestyle="dashdot",
        alpha=0.5
    )

ax.set_xlim([0, 20])
ax.get_xaxis().set_ticks([0, 5.1, 10.4, 14.4])

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos:f"{x}m" if x else 0))

# plt.legend()

plt.show()
