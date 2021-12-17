function [dist, ampl_coeff] = generate_point(start_point_m, cur_time, key_veloc_kmh, scenario_matrix, scenario_noise)

key_shift_m = key_veloc_kmh / 3600 * 1000 * cur_time;
x_path = start_point_m - key_shift_m;

% sensor and car height
y_height = 1;

% ceiling height
ceil_height = 5;

% back wall position
wall_pos_x = 13;
% wall_pos_x = 43;
% wall_pos_x = 80.5;

% car position
car_pos = 0;

dist(:, 1) = abs(car_pos - x_path);
dist(:, 2) = 2 * sqrt((dist(:, 1)/2).^2 + (y_height).^2);
dist(:, 3) = 2 * sqrt((dist(:, 1)/2).^2 + (ceil_height - y_height).^2);
dist(:, 4) = 2 * abs(car_pos - wall_pos_x) - dist(:, 1);
ampl_coeff = scenario_matrix;
ampl_coeff = repmat(ampl_coeff , size(dist, 1), 1);
ampl_coeff = ampl_coeff + scenario_noise * rand(size(ampl_coeff));

##printf("    gen_p dist:%s\n",mat2str(size(dist)))
##printf("    gen_p ampl_coeff:%s\n",mat2str(size(ampl_coeff)))

end
