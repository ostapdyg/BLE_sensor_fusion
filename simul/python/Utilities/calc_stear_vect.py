# function [G] = calc_stear_vect(omega_id, s_dist)

# c = physconst('LightSpeed');
# freq_low = 2.402e9; %2.402 GHz
# omega_low = 2.0 * pi * freq_low; % To radians
# omega_step = 2.0 * pi * 1e6; % 1 MHz in radians

# omega = omega_low + omega_id * omega_step;

# tQ = repmat(s_dist, length(omega), 1);
# tomega = repmat(omega', 1, length(s_dist));

# G = exp(-1i * tQ .* tomega / c);

# end
import numpy as np


def calc_stear_vect(freq_ids, dists) -> np.ndarray:
    """calculate ideal iqs

    Args:
        freq_ids (np.array(freq_nuum)): list of freq ids (0-40)
        dists (np.array(dist_num)): array of possible distances (m)
    """
    c = 299792458
    freq_0 = 2.402e9  # 2.402 GHz

    res = np.zeros([freq_ids.shape[0], dists.shape[0]])
    omegas = 2.0*np.pi*np.arange(freq_0, freq_0+freq_ids.shape[0]*1e6, 1e6)
    for freq_idx in range(np.shape(freq_ids)[0]):
        omega = 2.0*np.pi*(freq_0 + freq_ids[freq_idx]*1e6)
        # IQ = e^(-i*omega*t) = e^(-i*omega*l/c)
        res[freq_idx, :] = np.exp(-1j*dists*omega/c)
        # for dist_idx in range(np.shape(dists)[0]):
        #     res[freq_idx, dist_idx] = np.exp(-1j*dists[dist_idx]*omega/c)   # IQ = e^(-i*omega*t) = e^(-i*omega*l/c)
    return res
