import numpy as np
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker



from Src.plot import plot_format_axes
from Src.estimate_dist import fft


# plt.rcParams['text.usetex'] = True

x = np.arange(10, 50, 1)

freqs = {0.03823:1}
# x = np.sin()
y = np.zeros(x.shape)
for freq in freqs:
    y += freqs[freq]*np.sin(2*np.pi*(x+5)*freq)

fig, ax = plt.subplots()

ax.plot(x, y, "-", color="blue", alpha=0.7)

ax.axvline(x[0], 0, 1, color='black', ls='--', alpha=0.5)
ax.axvline(x[-1], 0, 1, color='black', ls='--', alpha=0.5)

plot_format_axes(ax)

# ax.xaxis.set_ticks([0, 1, 2, 5])
# ax.set_xlabel("$F_f$, s")
ax.text(1, 1, '$I(f,x)$')
ax.text(58, 0.07, "$f$, $s^{-1}$")

ax.text(x[0]+1, 0.5, '$f_0$')
ax.text(x[-1]+1, 0.5, '$f_0+\Delta f$')


ax.xaxis.set_ticks([])
# ax.xaxis.set_ticks([0, x[0], x[-1], 60])

# ax
ax.yaxis.set_ticks([])

plt.show()

# f.savefig("foo.pdf", bbox_inches='tight')