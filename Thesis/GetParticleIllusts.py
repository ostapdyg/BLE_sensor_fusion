import numpy as np
import scipy.stats
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patches

from Src.particle_filter import pf_resample_if_needed, pf_update_norm


N_PARTICLES = 100


PLOT_UPDATE = True
PLOT_PREDICT = True
PLOT_RESAMPLE = True

def main():

    fig, axes = plt.subplots(1,3)

    for ax in axes:
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])



    particles = np.empty((N_PARTICLES, 3))
    particles[:, 0] = np.random.normal(3, 1, size=N_PARTICLES)
    particles[:, 1] = np.random.normal(5., 1, size=N_PARTICLES)
    weights = np.ones((N_PARTICLES))

    pf_update_norm(weights, np.linalg.norm(particles[:, 0:2] - np.array([3,5]), axis=1), 0.4)

    if(PLOT_PREDICT):
        axes[0].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.2)
        axes[0].scatter(3, 5, alpha = 0.4, color="red", marker="x", s=100)

    particles[:,0] += np.random.normal(0, 5, size=N_PARTICLES)
    particles[:,1] += np.random.normal(0, 5, size=N_PARTICLES)

    if(PLOT_PREDICT):
        axes[0].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.5)
        axes[0].scatter(7, 5, alpha = 1, color="red", marker="x", s=100)
    if(PLOT_UPDATE):
        axes[1].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.2)

    pf_update_norm(weights, np.linalg.norm(particles[:, 0:2] - np.array([7,5]), axis=1), 0.4)


    if(PLOT_UPDATE):
        axes[1].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.7)
        axes[1].scatter(7, 5, alpha = 1, color="red", marker="x", s=100)
    if(PLOT_RESAMPLE):
        axes[2].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.2)
        # axes.scatter(7, 5, alpha = 1, color="red", marker="x", s=100) 

    pf_resample_if_needed(particles, weights, 1)

    if(PLOT_RESAMPLE):
        axes[2].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.7)
        axes[2].scatter(7, 5, alpha = 1, color="red", marker="x", s=100)

    axes[0].text(3, 1, "Predict")
    axes[1].text(3, 1, "Update")
    axes[2].text(3, 1, "Resample")

    fig.patches.append(patches.ConnectionPatch(
        (10.2, 5),
        (-0.2, 5),
        coordsA=axes[0].transData,
        coordsB=axes[1].transData,
        # Default shrink parameter is 0 so can be omitted
        # color="black",
        arrowstyle="->",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=2,
    ))

    fig.patches.append(patches.ConnectionPatch(
        (10.2, 5),
        (-0.2, 5),
        coordsA=axes[1].transData,
        coordsB=axes[2].transData,
        # Default shrink parameter is 0 so can be omitted
        # color="black",
        arrowstyle="->",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        linewidth=2,
    ))
    plt.show()


main()