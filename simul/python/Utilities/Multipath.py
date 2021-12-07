import numpy as np

# function [hs] = Multipath(omega, dist, ampl_coeff)
def Multipath(omega:float, dist:np.ndarray, ampl_coeff:np.ndarray)->float:
    # c = physconst('LightSpeed');
    # multipaths = zeros( length(omega),length(dist) );
    c = 3.0e9
    multipaths = np.zeros( np.size(omega), np.size(dist))

    # for f = 1:length(omega)
    # for f in range(np.size(omega)): Omega is float
    #     for k = 1:length(dist)
    for k in range(np.size(dist)):
#         a = c / (2 * dist(k) * omega(f)); % amplitude from distance
            a = c / (2 * dist[k] * omega) # amplitude from distance
    #         phi = dist(k) * omega(f) / c ; % phase from distance
            phi = dist[k] * omega / c # phase from distance
    #         multipaths(f,k) = ampl_coeff(k) * a * exp( - 1i * phi );
            multipaths[0,k] = ampl_coeff[k] * a * np.exp( - 1j * phi )
    #     end
    # end
    # hs = sum(multipaths, 2);
    hs = np.sum(multipaths, 1)
    # hs=hs.';
    return hs.transpose()
    # end


