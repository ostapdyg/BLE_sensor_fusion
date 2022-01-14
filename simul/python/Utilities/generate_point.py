import numpy as np
import numpy.matlib

from Utilities.parameters import Parameters


# @njit
def generate_point(
    p:Parameters, ts
) -> tuple[np.ndarray, np.ndarray]:
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
    dist = np.array(
        [
            [
                d0,
                2 * ((d0 / 2) ** 2 + (y_height) ** 2) ** 0.5,
                2 * ((d0 / 2) ** 2 + (ceil_height - y_height) ** 2) ** 0.5,
                2 * abs(car_pos - wall_pos_x) - d0,
            ]
        ]
    )
    # ampl_coeff = scenario_matrix;
    # ampl_coeff = repmat(ampl_coeff , size(dist, 1), 1);
    # ampl_coeff = ampl_coeff + scenario_noise * rand(size(ampl_coeff));
    ampl_coeff = np.array(p.scenario_matrix)

    # ampl_coeff = np.expand_dims(np.array(scenario_matrix).repeat(dist.shape[0]), axis=1)
    ampl_coeff = np.matlib.repmat(ampl_coeff, dist.shape[0], 1)
    ampl_coeff = ampl_coeff + p.scenario_noise * np.random.uniform(0, 1, ampl_coeff.shape)

    return (dist, ampl_coeff)
