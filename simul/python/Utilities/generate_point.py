from typing import Tuple
import numpy as np
import numpy.matlib

# function [dist, ampl_coeff] = generate_point(start_point_m, cur_time, key_veloc_kmh, scenario_matrix, scenario_noise)
def generate_point(start_point_m, cur_time, key_veloc_kmh, scenario_matrix, scenario_noise)->tuple[np.ndarray, np.ndarray]:
    # key_shift_m = key_veloc_kmh / 3600 * 1000 * cur_time;
    key_shift_m = key_veloc_kmh / 3600 * 1000 * cur_time
    # x_path = start_point_m - key_shift_m;
    x_path = start_point_m - key_shift_m
    # % sensor and car height
    # y_height = 1;
    y_height = 1

    # % ceiling height
    # ceil_height = 5;
    ceil_height = 5
    # % back wall position
    # wall_pos_x = 13;
    wall_pos_x = 13
    # % wall_pos_x = 43;
    # % wall_pos_x = 80.5;

    # % car position
    # car_pos = 0;
    car_pos = 0
    # d0 = abs(car_pos - x_path);
    # dist(:, 2) = 2 * sqrt((d0/2).^2 + (y_height).^2);
    # dist(:, 3) = 2 * sqrt((d0/2).^2 + (ceil_height - y_height).^2);
    # dist(:, 4) = 2 * abs(car_pos - wall_pos_x) - d0;
    d0 = abs(car_pos - x_path)
    dist = np.array([[
        d0,
        2 * ((d0/2)**2 + (y_height)**2)**.5,
        2 * ((d0/2)**2 + (ceil_height - y_height)**2)**.5,
        2 * abs(car_pos - wall_pos_x) - d0
    ]])
    # ampl_coeff = scenario_matrix;
    # ampl_coeff = repmat(ampl_coeff , size(dist, 1), 1);
    # ampl_coeff = ampl_coeff + scenario_noise * rand(size(ampl_coeff));
    ampl_coeff = np.array(scenario_matrix)

    ampl_coeff = np.matlib.repmat(ampl_coeff , dist.shape[0], 1)
    ampl_coeff = ampl_coeff + scenario_noise * np.random.uniform(0, 1, ampl_coeff.shape)
    # print(f"    gen_p dist:{dist.shape}")
    # print(f"    gen_p ampl_coeff:{ampl_coeff.shape}")

    return (dist, ampl_coeff)


# end
if(__name__ == "__main__"):
    print( generate_point(10, 0, 5, [0.2, 0, 0, 0.6], 0))