function [hs] = Multipath(omega, dist, ampl_coeff)

% pkg load miscellaneous;
% c = physical_constant('speed of light in vacuum');

% c = physconst('LightSpeed');
c = 299792458;

multipaths = zeros( length(omega),length(dist) );

for f = 1:length(omega)
    for k = 1:length(dist)
        a = c / (2 * dist(k) * omega(f)); % amplitude from distance
        phi = dist(k) * omega(f) / c ; % phase from distance
        multipaths(f,k) = ampl_coeff(k) * a * exp( - 1i * phi );
    end
end

hs = sum(multipaths, 2);
hs=hs.';
end


