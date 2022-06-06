import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Src.estimate_dist import fft
from Src.plot import plot_format_axes

# plt.rcParams['text.usetex'] = True

x = np.arange(0, 50, 0.01)
freqs = {1:1, 2:0.5, 5:0.7}
# x = np.sin()
y = np.zeros(x.shape)
for freq in freqs:
    y += freqs[freq]*np.sin(2*np.pi*x*freq)


fft_amps, fft_freqs = fft(y, x)
fig, ax = plt.subplots()

ax.plot(fft_freqs[(fft_freqs>=0)*(6>=fft_freqs)], fft_amps[(fft_freqs>=0)*(6>=fft_freqs)], "-")

plot_format_axes(ax)

ax.text(6, 100, "$F_f$, s")

ax.xaxis.set_ticks([0, 1, 2, 5])
ticks = {0:'0', 1:'$x_0/c$', 2:'$x_1/c$', 5:'$x_2/c$'}
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,pos: ticks[x] if x in ticks else ''))
ax.yaxis.set_ticks([])
plt.show()

# f.savefig("foo.pdf", bbox_inches='tight')