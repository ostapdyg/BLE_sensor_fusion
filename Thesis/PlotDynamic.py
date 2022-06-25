

import numpy as np
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
import matplotlib.pyplot as plt

from Src.simul import simul_get_pos, simul_signals_shift_full
from Src.simul import ANCHOR_COORDS, TARGET_START, TARGET_TURNS

from Src.estimate_dist import estimate_distance

from Src.particle_filter import pf_generate_norm, pf_predict_rot, pf_predict_mov, pf_update_norm, pf_resample_if_needed

from Src.plot import plot_positions, plot_errors


BASELINE_VEL = 1.0

USE_IMU = True

PARTICLE_N = 5000

MEASURES_PER_S = 3

def main():
    N = 5000
    np.random.seed(123)


    particles = pf_generate_norm(
            mean=(0, 0, 0), std=(5, 5, np.pi/4), N=PARTICLE_N)

    weights = np.ones(PARTICLE_N) / PARTICLE_N

    tss = np.linspace(0, 12, MEASURES_PER_S*12 + 1) # 3 per s
    xss = np.array([simul_get_pos(t) for t in tss])

    xss_predicted = np.zeros([tss.shape[0], 2])


    unused_b_start = 0
    unused_b_len = 0

    for i_ts in range(4):
        # i_anchor = i_ts % 4
        
        # Predict
        pf_predict_mov(particles, (BASELINE_VEL + np.random.randn()*0.2)*(tss[1]-tss[0]), 0.2)
        if(USE_IMU):
            if(i_ts in TARGET_TURNS):
                pf_predict_rot(particles, TARGET_TURNS[i_ts] + np.random.randn()*(0.5), 0.4)
        pf_predict_rot(particles, 0, 0.2)

        # BLE
        curr_anchor = ANCHOR_COORDS[i_ts%4]
        
        signal_vals = simul_signals_shift_full(curr_anchor, xss[i_ts])
        # Unused signals...
        if(i_ts%(2*MEASURES_PER_S)):
            unused_b_start = np.random.randint(5, 20)
            unused_b_len = np.random.randint(5, 10)
        signal_vals[unused_b_start:unused_b_start+unused_b_len] = np.nan

        ble_dist = estimate_distance(signal_vals)

        particle_dist_2d = np.linalg.norm(particles[:, 0:2] - curr_anchor[0:2], axis=1) # XY
        particle_dist = np.sqrt(np.square(particle_dist_2d) + np.square(curr_anchor[2] - 1)) # Z

        # Update
        pf_update_norm(weights, ble_dist - particle_dist, 0.4)

        # Resample
        pf_resample_if_needed(particles, weights, 0.5)

        # Estimate
        xss_predicted[i_ts,:] = np.average(particles[:,:2], weights=weights, axis=0)
        # print("  err1:",np.sqrt(np.square(xss_predicted[i_ts,:2] - xss[i_ts,:2]).mean()))
        print("    err:",np.linalg.norm(xss_predicted[i_ts, 0:2] - xss[i_ts, 0:2], axis=0))


    # print("Error:",np.sqrt(np.square(xss_predicted[2:,:2] - xss[2:,:2]).mean()))
    print("Error:",np.sqrt(np.square(np.linalg.norm(xss_predicted[2:, 0:2] - xss[2:, 0:2], axis=1))).mean())

    fig_pos, ax = plot_positions(xss, xss_predicted, ANCHOR_COORDS, 6, 5)
    fig_err, ax = plot_errors(xss_predicted[2:,:2] - xss[2:,:2], 1, 1)

    plt.show()
    # plotly.io.write_image(fig2, 'error.pdf', format='pdf')


main()