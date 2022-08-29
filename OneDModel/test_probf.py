from cProfile import label
import numpy as np
import pandas as pd


from Src.imu_model import IMUModel

from Src.movementModel import MovementModel


from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

PARTICLE_N = 500

MEASURES_PER_S = 5
T_MAX = 5


def get_ble_vels(mov, t):
    WINDOW_SIZE = 0.2
    WINDOW_STEPS = 100
    # t += 2
    tss = t + np.linspace(-WINDOW_SIZE, 0, WINDOW_STEPS)
    xss = np.array([mov.get_pos(ts) for ts in tss])
    # print(f"  tss:{tss.shape}, xss:{xss}")
    # res = 
    # for freq in FREQS:
    freq = FREQS[0]
    signal_vals = np.array([simul_signals_shift_freq(xs, freq) for xs in xss])
    # plot_pdf(signal_vals.real)
    # plt.show()
    # print(signal_vals.shape, signal_vals)
    return estimate_speed_probf(signal_vals, tss, freq)
    
    

def main():
    mov = MovementModel()
    # imu = IMUModel("IMU_data.csv")

    np.random.seed(123)

    particles = pf_generate_uniform(
        x_range=(0, 10), vel_range=(0, 0), rot_range=(0, 0.2), N=PARTICLE_N)

    weights = np.ones(PARTICLE_N) / PARTICLE_N

    tss = np.linspace(0, T_MAX, MEASURES_PER_S*T_MAX + 1)  # 3 per s
    # xss = np.array([mov(t) for t in tss])
    dist_ble = np.zeros(tss.shape, dtype=np.float64)
    dist_particle = np.zeros(tss.shape, dtype=np.float64)
    speed_particle = np.zeros(tss.shape, dtype=np.float64)
    speed_imu = np.zeros(tss.shape, dtype=np.float64)
    speed_true = np.zeros(tss.shape, dtype=np.float64)
    particle_rot = np.zeros(tss.shape, dtype=np.float64)

    xss_prev = 0
    xss = np.zeros((*tss.shape, 2))

    particle_dists_full = np.zeros((tss.shape[0], particles.shape[0]))
    particle_vels_full = np.zeros((tss.shape[0], particles.shape[0]))
    particle_weights_full = np.zeros((tss.shape[0], particles.shape[0]))

    prob_xss = np.arange(0, 6, 0.01)
    ble_probs = np.zeros((tss.shape[0], prob_xss.shape[0]))

    for i_ts in range(tss.shape[0]):
        dt = 1.0/MEASURES_PER_S

        print(f"{tss[i_ts]:.2f}s:")
        # xss[i_ts] = mov.update(tss[i_ts], dt)
        xss[i_ts] = np.array([2.0, 0.0])


        speed = (xss[i_ts] - xss_prev)/dt
        xss_prev = xss[i_ts]
        speed_true[i_ts] = (speed[0]**2+speed[1]**2)**0.5
        # print(tss[i_ts], np.round(xss[i_ts], 4))
        

        # IMU
        # imu.update(tss[i_ts], dt)
        # imu_speed = imu.get_speed()
        # imu_rot = imu.get_rot()

        # Predict
        # pf_predict_mov(particles, 0, 0.05)
        # pf_predict_mov(particles, dt, 0.05)
        # pf_predict_vel(particles, 0.0, 0.2)
        # pf_predict_rot(particles, 0.0, 0.01)

        # if (imu_rot > 10):
        # pf_predict_rot(particles, imu_rot*(np.pi/180), 0.0)

        # Update
        # BLE

        signal_vals = simul_signals_shift_full(np.array([1e-100, 1e-100]), xss[i_ts])
        #  + \
                # simul_signals_shift_full(np.array([1e-100, 1e-100]), np.array([0, 4]))
        # probs = fft_f(particle_dist)
        # pf_update_distr(weights, probs)
        # 
        
        # ble_dist = estimate_distance(signal_vals)
        # prob_f = lambda x:scipy.stats.norm(0, 0.1).pdf(ble_dist - x)
        prob_f = estimate_dist_probf(signal_vals)

        ble_probs[i_ts, :] = prob_f(prob_xss)
        pf_update_distr(weights, prob_f(particles[:, 0]))

        # pf_update_norm(weights, ble_dist - particles[:, 0], 0.1)



        particle_weights_full[i_ts,:] = weights
        particle_dists_full[i_ts,:] = particles[:, 0]
        particle_vels_full[i_ts,:] = particles[:, 1]

        # Resample
        pf_resample_if_needed(particles, weights, 0.5)

    # signal_vals = simul_signals_shift_full(np.array([1e-100, 1e-100]), xss[0])
    # prob_f = estimate_dist_probf(signal_vals)
    # ble_probs[0, :] = prob_f(prob_xss)



    dist_true = (xss[:, 0]**2 + xss[:, 1]**2)**.5
    ax = plot_particles_scatter(particle_dists_full, particle_weights_full)
    # plt.axis('scaled')
    ax.set_ylim([-.05, 6.05])
    ax.plot(range(tss.shape[0]), dist_true, c="g")
    # plot_particles_color(particle_dists_full[1:], particle_weights_full[1:])

    # fig, ax = plt.subplots()
    # fig, (ax_graph, ax_im) = plt.subplots(1, 2)
    fig, (ax_graph, ax_im) = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1,5],})

    ax_graph.grid(True)
    ax_graph.plot(ble_probs[0, :], prob_xss)
    ax_graph.invert_xaxis()


    # fig, ax = plt.subplots()
    ax_im.grid(True)
    ax_im.axes.get_xaxis().set_ticks([])
    ax_im.axes.get_yaxis().set_ticks([])

    p = ax_im.imshow(ble_probs.T[::-1, :], aspect="auto", cmap="inferno", # viridis
                  extent=[0, tss.shape[0], 0, 6, ])
    ax_im.plot(range(tss.shape[0]), dist_true, c="g")
    fig.colorbar(p)
    fig.tight_layout()

    plt.show()
    # plotly.io.write_image(fig2, 'error.pdf', format='pdf')


main()
