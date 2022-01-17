function [G] = calc_stear_vect(omega_id, s_dist)

% c = physconst('LightSpeed');
c = 299792458;
freq_low = 2.402e9; %2.402 GHz
omega_low = 2.0 * pi * freq_low; % To radians
omega_step = 2.0 * pi * 1e6; % 1 MHz in radians

omega = omega_low + omega_id * omega_step;

tQ = repmat(s_dist, length(omega), 1);
tomega = repmat(omega', 1, length(s_dist));

G = exp(-1i * tQ .* tomega / c);

end

