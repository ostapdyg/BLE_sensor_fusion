import numpy as np

# function [hs] = Multipath(omega, dist, ampl_coeff)
def Multipath(omega:float, dist:np.ndarray, ampl_coeff:np.ndarray)->float:
    # c = physconst('LightSpeed');
    # multipaths = zeros( length(omega),length(dist) );
    c = 3.0e9
#     print(f"    Multipath omega:{omega.shape or 1}")
#     print(f"    Multipath dist:{dist.shape}")
#     print(f"    Multipath ampl_coeff:{ampl_coeff.shape}")

    multipaths = np.zeros( [1, max(dist.shape)], dtype=np.csingle)
#     print(f"    Multipath multipaths:{multipaths.shape}")

    # for f = 1:length(omega)
    # for f in range(np.size(omega)): Omega is float
    #     for k = 1:length(dist)
    for k in range(np.size(dist)):
#         a = c / (2 * dist(k) * omega(f)); % amplitude from distance
        a = c / (2 * dist[0,k] * omega) # amplitude from distance
        #         phi = dist(k) * omega(f) / c ; % phase from distance
        phi = dist[0, k] * omega / c # phase from distance
        # print(f"    Multipath dist:{phi.shape}")
        # print(f"    Multipath a:{a.shape}")

        #         multipaths(f,k) = ampl_coeff(k) * a * exp( - 1i * phi );
        multipaths[:,k] = ampl_coeff[0,k] * a * np.exp( - 1j * phi )
#     print(f"    Multipath mp:{multipaths}")
    #     end
    # end
    # hs = sum(multipaths, 2);
    hs = np.sum(multipaths, 1)
    # hs=hs.';
    return hs.transpose()
    # end


