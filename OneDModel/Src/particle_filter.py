import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import scipy.stats


# particles: dist, vel, anrotgle

def pf_generate_uniform(x_range, vel_range, rot_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(vel_range[0], vel_range[1], size=N)
    particles[:, 2] = np.random.uniform(rot_range[0], rot_range[1], size=N)
    # particles[:, 2] %= 2 * np.pi
    return particles

# def pf_generate_norm(mean, std, N):
#     particles = np.empty((N, 3))
#     particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
#     particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
#     particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
#     particles[:, 2] %= 2 * np.pi
#     return particles

def pf_predict_rot(particles, d_rot, std=.0):
    particles[:, 2] += d_rot + (np.random.randn(particles.shape[0]) * std)
    particles[:, 2] %= 2 * np.pi

def pf_predict_vel(particles, d_vel, std=.0):
    particles[:, 1] += d_vel + (np.random.randn(particles.shape[0]) * std)
    return

def pf_predict_mov(particles, dt, std=.0):
    dist = particles[:, 1]*dt + (np.random.randn(particles.shape[0]) * std)
    particles[:, 0] += np.cos(particles[:, 2]) * dist

# Deltas - diff between expected and received
def pf_update_norm(weights, deltas, std):
    weights *= scipy.stats.norm(0, std).pdf(deltas)
    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def pf_update_distr(weights, probs):
    # weights *= 
    # for (x in xs): 
    weights *= probs
    weights += 1.e-100      # avoid round-off to zero
    weights /= sum(weights) # normalize


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""
    mean = np.average(particles, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def effective_N(weights):
    return 1. / np.sum(np.square(weights))

# make N subdivisions, choose positions 
# with a consistent random offset
def systematic_resample(particles, weights):
    N = len(weights)

    positions = (np.arange(N) + np.random.rand()) / N

    resampled_i = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            resampled_i[i] = j
            i += 1
        else:
            j += 1
    particles[:] = particles[resampled_i]
    weights.resize(len(particles))
    # weights.fill(1.0 / len(weights))
    weights[:] = 1.0 / len(weights)

def pf_resample_if_needed(particles, weights, threshold=0.5, method="systematic"):
    if(effective_N(weights) < particles.shape[0]*threshold):
        print("  resample...")
        systematic_resample(particles, weights)
        pf_predict_mov(particles, 0, 0.01)


# from  import plot_pdf
if (__name__ == "__main__"):
    PARTICLE_N = 500
    particles = pf_generate_uniform(
        x_range=(0, 10), vel_range=(0, 3), rot_range=(0, 0.2), N=PARTICLE_N)

    weights = np.ones(PARTICLE_N) / PARTICLE_N

    probs = scipy.stats.norm(5, 100).pdf(np.arange(0, 100000))
    plot_pdf(probs).show()
    # print(probs)
    pf_update_distr(weights, particles[:, 0], probs)
    print(weights)
