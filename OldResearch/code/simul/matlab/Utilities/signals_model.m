function [r, signals] = signals_model(omega_id, dist, ampl_coeff, delays, noises)

freq_low = 2.402e9; %2.402 GHz
% freq_high = 2.480e9; %2.480 GHz
omega_low = 2.0 * pi * freq_low; % To radians
omega_step = 2.0 * pi * 1e6; % 1 MHz in radians

omega = omega_low + omega_id * omega_step;
hs = Multipath(omega, dist, ampl_coeff);

if isempty(delays)
    delays = 1;
end
if isempty(noises)
    noises = 0;
end

signals = delays .* abs(hs) .* exp(1i * 2 * angle(hs)) + noises;

signals = signals.';
r = signals*signals';

end
