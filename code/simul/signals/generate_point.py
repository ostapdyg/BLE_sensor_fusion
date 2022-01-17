import numpy as np

from simul.parameters import Parameters


def generate_point(p: Parameters, ts) -> tuple[np.ndarray, np.ndarray]:
    key_shift_m = (p.vel / 3600) * 1000 * ts
    x_path = p.start_pos - key_shift_m
    # sensor and car height
    y_height = 1

    # % ceiling height
    ceil_height = 5
    # % back wall position
    wall_pos_x = 13
    # % wall_pos_x = 43;
    # % wall_pos_x = 80.5;

    # % car position
    car_pos = 0
    d0 = abs(car_pos - x_path)
    #  TODO: Figure out why d0 not multiplied by 2 <21-12-21, astadnik> #
    dist = [
        d0,
        2 * ((d0 / 2) ** 2 + (y_height) ** 2) ** 0.5,
        2 * ((d0 / 2) ** 2 + (ceil_height - y_height) ** 2) ** 0.5,
        2 * abs(car_pos - wall_pos_x) - d0,
    ]
    dist = np.expand_dims(np.array(dist), axis=0)
    # ampl_coeff = scenario_matrix;
    # ampl_coeff = repmat(ampl_coeff , size(dist, 1), 1);
    # ampl_coeff = ampl_coeff + scenario_noise * rand(size(ampl_coeff));
    ampl_coeff = np.array(p.scenario_matrix)

    ampl_coeff = np.expand_dims(
        np.array(p.scenario_matrix).repeat(dist.shape[0]), axis=0
    )
    ampl_coeff = ampl_coeff + p.scenario_noise * np.random.uniform(
        0, 1, ampl_coeff.shape
    )

    return (dist, ampl_coeff)
