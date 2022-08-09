from cProfile import label
import numpy as np
import pandas as pd


from Src.imu_model import IMUModel

from Src.movementModel import MovementModel


from Src.particle_filter import *
from Src.plot import *
from Src.ble_model import *

USE_IMU = True

PARTICLE_N = 200

MEASURES_PER_S = 5
T_MAX = 12


def plot_particles_scatter(particles, weights):
    fig, ax = plt.subplots()
    ax.grid(True)
    # dists = np.array([i for ])
    for i in range(particles.shape[0]):
        # kde = scipy.stats.gaussian_kde(xs[i,:], weights=weights[i, :])
        # ax.plot(dists, kde(dists))
        # x = xs[i]
        ax.scatter(i + (np.random.random(particles.shape[1])-0.5) * 0.7, 
                    particles[i, :], s=(1 + weights[i, :]*PARTICLE_N*2))
        # , color = "green" if (i%2) else "blue")
    return ax



def main():
    mov = MovementModel()
    imu = IMUModel("IMU_data.csv")
    N = 500000
    np.random.seed(123)

    particles = pf_generate_uniform(
        x_range=(0, 10), vel_range=(0, 3), rot_range=(0, 0.2), N=PARTICLE_N)

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

    dists_full = np.zeros((tss.shape[0], particles.shape[0]))
    vels_full = np.zeros((tss.shape[0], particles.shape[0]))
    weights_full = np.zeros((tss.shape[0], particles.shape[0]))

    for i_ts in range(tss.shape[0]):
        dt = 1.0/MEASURES_PER_S



        # i_anchor = i_ts % 4
        print(f"{tss[i_ts]:.2f}s:")
        xss[i_ts] = mov.update(tss[i_ts], dt)
       
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
        pf_predict_vel(particles, 0.0, 0.4)
        pf_predict_rot(particles, 0.0, 0.01)
        # pf_predict_rot(particles, 0.0, 0.05)

        # if (imu_rot > 10):
        pf_predict_rot(particles, imu_rot*(np.pi/180), 0.0)

        # BLE
        signal_vals = simul_signals_shift_full(np.array([1e-100, 1e-100]), xss[i_ts])

        ble_dist = estimate_distance(signal_vals)
        dist_ble[i_ts] = ble_dist


        speed_imu[i_ts] = imu.get_speed()
        # Update

        # fft_f = estimate_dist_probf(signal_vals)
        # probs = fft_f(particle_dist)
        # pf_update_distr(weights, probs)
        # 
        pf_update_norm(weights, ble_dist - particles[:, 0], 0.1)
        # pf_update_norm(weights, speed_imu[i_ts] - particles[:, 1], 0.3)

        weights_full[i_ts,:] = weights
        dists_full[i_ts,:] = particles[:, 0]
        vels_full[i_ts,:] = particles[:, 1]
        # Resample
        pf_resample_if_needed(particles, weights, 0.5)

        # Estimate
        # dist_particle[i_ts] = np.average(
        #     particles[:, 0], weights=weights, axis=0)
        # speed_particle[i_ts] = np.average(
        #     particles[:, 1], weights=weights, axis=0)
        # particle_rot[i_ts] = np.average(
        #     particles[:, 2], weights=weights, axis=0)

        # print(f"  imu_rot:{imu_rot:.2f}; rot:{np.average(particles[:, 2], weights=weights, axis=0)*180/np.pi:.2f}")
        # print(f"  imu:{imu_speed:.2f}m/s; part:{speed_particle[i_ts]:.2f}m/s")
        # print(f"  ble:{ble_dist:.2f}; part:{dist_particle[i_ts]:.2f}")
        # print("  err1:",np.sqrt(np.square(xss_predicted[i_ts,:2] - xss[i_ts,:2]).mean()))
        # print("    err:",np.linalg.norm(xss_predicted[i_ts, 0:2] - xss[i_ts, 0:2], axis=0))

    # print("Error:",np.sqrt(np.square(xss_predicted[2:,:2] - xss[2:,:2]).mean()))
    # print("Error:",np.sqrt(np.square(np.linalg.norm(xss_predicted[2:, 0:2] - xss[2:, 0:2], axis=1))).mean())

    dist_true = (xss[:, 0]**2 + xss[:, 1]**2)**.5
    ax = plot_particles_scatter(dists_full, weights_full)
    ax.plot(range(tss.shape[0]), dist_true)
    plot_particles_scatter(vels_full, weights_full)

    # fig_pos, ax = plot_positions(xss, xss_predicted, ANCHOR_COORDS, 6, 5)
    # fig_pos, ax = plot_dist(dist_true,
    #                         {
    #                             "ble_only": dist_ble,
    #                             "particle": dist_particle,
    #                         }, tss)

    # fig_pos, ax = plot_dist(speed_true,
    #                         {
    #                             "imu": speed_imu,
    #                             "particle": speed_particle,
    #                             "rot":particle_rot/np.pi

    #                         }, tss)

    # fig_pos, ax = plot_dist(180*particle_rot/np.pi,
    #                     {
    #                     }, tss)
    # fig_err, ax = plot_errors(xss_predicted[2:,:2] - xss[2:,:2], 1, 1)

    # fig_err2, ax = plot_abserr(xss_predicted[0:,:2] - xss[0:,:2], tss)

    plt.show()
    # plotly.io.write_image(fig2, 'error.pdf', format='pdf')


main()
