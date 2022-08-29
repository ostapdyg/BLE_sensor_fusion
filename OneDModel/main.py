from cProfile import label
import numpy as np
import pandas as pd


from Src.imu_model import IMUModel

from Src.movementModel import MovementModel


from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

PARTICLE_N = 200000

MEASURES_PER_S = 5
T_MAX = 9

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
    imu = IMUModel("IMU_data.csv")

    np.random.seed(123)

    particles = pf_generate_uniform(
        # x_range=(0, 10), vel_range=(0, 3), rot_range=(0, 0.2), N=PARTICLE_N)
        x_range=(0, 10), vel_range=(0, 3), rot_range=(-np.pi, np.pi), N=PARTICLE_N)

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
    particle_rots_full = np.zeros((tss.shape[0], particles.shape[0]))
    # signal_vals_full = np.zeroes((tss.shape[0], 40))
    vel_steps = np.arange(-5, 5, 0.1)
    vel_probs_full = np.zeros((tss.shape[0], vel_steps.shape[0]))


    for i_ts in range(tss.shape[0]):
        dt = 1.0/MEASURES_PER_S



        # i_anchor = i_ts % 4
        print(f"{tss[i_ts]:.2f}s:")
        # xss[i_ts] = mov.update(tss[i_ts], dt)
        xss[i_ts] = mov.get_pos(tss[i_ts])

        speed = (xss[i_ts] - xss_prev)/dt
        xss_prev = xss[i_ts]
        speed_true[i_ts] = (speed[0]**2+speed[1]**2)**0.5
        # print(tss[i_ts], np.round(xss[i_ts], 4))


        # IMU
        imu.update(tss[i_ts], dt)
        imu_speed = imu.get_speed()
        imu_rot = imu.get_rot()

        # Predict
        pf_predict_mov(particles, dt, 0.2)
        pf_predict_vel(particles, 0.0, 1.0)
        # pf_predict_rot(particles, 0.0, 1.0)

        # if (imu_rot > 10):
        pf_predict_rot(particles, imu_rot*(np.pi/180), 1.0)

        # BLE
        signal_vals = simul_signals_shift_full(np.array([1e-100, 1e-100]), xss[i_ts])

        # ble_dist = estimate_distance(signal_vals)
        # dist_ble[i_ts] = ble_dist

        vel_f = get_ble_vels(mov, tss[i_ts])
        vel_probs_full[i_ts, :] = vel_f(vel_steps)
        vel_probs_full[i_ts, :] = vel_probs_full[i_ts, :]/np.sum(vel_probs_full[i_ts, :])
        probs = vel_f(-particles[:, 1]*np.cos(particles[:,2]))
        pf_update_distr(weights, probs)

        # Update

        fft_f = estimate_dist_probf(signal_vals)
        probs = fft_f(particles[:, 0])
        pf_update_distr(weights, probs)

        # pf_update_norm(weights, ble_dist - particles[:, 0], 0.1)
        
        speed_imu[i_ts] = imu.get_speed()
        # pf_update_norm(weights, speed_imu[i_ts] - particles[:, 1], 0.1)
        pf_update_norm(weights, speed_true[i_ts] - particles[:, 1], 0.1)



        particle_weights_full[i_ts,:] = weights
        particle_dists_full[i_ts,:] = particles[:, 0]
        particle_vels_full[i_ts,:] = particles[:, 1]
        particle_rots_full[i_ts,:] = particles[:, 2]
        # Resample
        pf_resample_if_needed(particles, weights, 0.5)

    dist_true = (xss[:, 0]**2 + xss[:, 1]**2)**.52
    # rot_true
    ax = plot_particles_scatter(particle_dists_full, particle_weights_full)
    ax.plot(range(tss.shape[0]), dist_true, c="g")
    ax = plot_particles_scatter(particle_vels_full, particle_weights_full)
    ax.plot(range(tss.shape[0]), speed_true, c="g")
    ax = plot_particles_scatter(particle_vels_full*np.cos(particle_rots_full), particle_weights_full)
    # ax = plot_particles_color(particle_vels_full*np.cos(particle_rots_full), particle_weights_full)
    # ax.plot(range(tss.shape[0]), speed_true*np.cos(), c="g")

    ax = plot_particles_scatter((particle_rots_full/np.pi)*180, particle_weights_full)
    # ax.plot(range(tss.shape[0]), speed_true, c="g")

    # fig, ax = plt.subplots()
    # ax.imshow(vel_probs_full[1::, :], aspect="auto", cmap="viridis",
    #              )

    # fig, ax = plt.subplots()
    # ax.set_xlim([-5.5, 5.5])
    # ax.set_ylim([0, 1])

    #         # print(vel_probs_full[i_ts, :])
    #     ax.plot(vel_steps, vel_probs_full[i_ts2, :], label=f"{i_ts2}")
    # ax.legend()
    # plt.grid(True)

    fig, ax = plt.subplots()
    ax.imshow(vel_probs_full.T[::, :], aspect="auto", cmap="inferno", # viridis
                  extent=[0, tss.shape[0], 0, 100, ])

    # plot_pdf(vel_probs_full[10, :])
    # ax.plot(range(tss.shape[0]), ro, c="g")
    # plot_particles_color(particle_dists_full, particle_weights_full)
    # plot_particles_color(particle_vels_full[1:], particle_weights_full[1:])
    # plot_particles_color((particle_rots_full/np.pi)*180, particle_weights_full)


    plt.show()
    # plotly.io.write_image(fig2, 'error.pdf', format='pdf')


main()
