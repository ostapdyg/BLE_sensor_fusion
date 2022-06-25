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
PLOT_I = 0

def plot_particles(particles, weights, particles_new, weights_new, pos, pos_new):
    fig, ax = plt.subplots()

    plt.axis('scaled')

    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    ax.scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.2)
    ax.scatter(pos[0], pos[1], alpha = 0.4, color="red", marker="x", s=100)

    ax.scatter(particles_new[:,0], particles_new[:,1], s=2+(weights_new**.5)*N_PARTICLES*5, alpha=0.5)
    ax.scatter(pos_new[0], pos_new[1], alpha = 1, color="red", marker="x", s=100)

    global PLOT_I
    plt.savefig(f'plot_gif/plot{PLOT_I}.png', bbox_inches='tight')
    PLOT_I += 1


def main():

    pos = np.array([1, 5])
    np.random.seed(2123)
    particles = np.empty((N_PARTICLES, 3))
    particles[:, 0] = np.random.normal(pos[0], 1, size=N_PARTICLES)
    particles[:, 1] = np.random.normal(pos[1], 1, size=N_PARTICLES)
    weights = np.ones((N_PARTICLES))

    pf_update_norm(weights, np.linalg.norm(particles[:, 0:2] - pos, axis=1), 0.4)

    for i in range(3):
        particles_old = np.array(particles)
        weights_old = np.array(weights)
        pos_old = np.array(pos)

        pos += np.array([2, 0])
        particles[:,0] += 3 + np.random.normal(0, 0.5, size=N_PARTICLES)
        particles[:,1] += np.random.normal(0, 0.5, size=N_PARTICLES)
        plot_particles(particles_old, weights_old, particles, weights, pos_old, pos)

        particles_old = np.array(particles)
        weights_old = np.array(weights)
        pos_old = np.array(pos)

        pf_update_norm(weights, np.linalg.norm(particles[:, 0:2] - pos, axis=1), 0.4)

        plot_particles(particles_old, weights_old, particles, weights, pos_old, pos)

        # if(PLOT_UPDATE):
        #     axes[1].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.7)
        #     axes[1].scatter(7, 5, alpha = 1, color="red", marker="x", s=100)
        # if(PLOT_RESAMPLE):
        #     axes[2].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.2)
        #     # axes.scatter(7, 5, alpha = 1, color="red", marker="x", s=100) 

        particles_old = np.array(particles)
        weights_old = np.array(weights)
        pos_old = np.array(pos)

        pf_resample_if_needed(particles, weights, 1)
        particles[:,0] += np.random.normal(0, 0.1, size=N_PARTICLES)
        particles[:,1] += np.random.normal(0, 0.1, size=N_PARTICLES)

        plot_particles(particles_old, weights_old, particles, weights, pos_old, pos)

    # if(PLOT_RESAMPLE):
    #     axes[2].scatter(particles[:,0], particles[:,1], s=2+(weights**.5)*N_PARTICLES*5, alpha=0.7)
    #     axes[2].scatter(7, 5, alpha = 1, color="red", marker="x", s=100)

    plt.show()


main()