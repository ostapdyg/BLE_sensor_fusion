from numba import njit
import numpy as np

# from icecream import ic

# # function [spec, spec_1] = my_correlation_use(sv, signal, r)
# def my_correlation_use(sv, signal, r)->tuple[np.array, np.array]:
# # % spec = abs(sv' * signal).^2;

# # spec = zeros(size(sv, 2), 1);
#     spec = np.zeros((np.size(sv, 1), 1))
# # for idx = 1:size(sv, 2)
#     for idx in range(np.size(sv, 1)):
# #     spec(idx) = abs(sv(:, idx)' * r * sv(:, idx));
#         spec[idx] = np.abs(ctranspose(sv[:,idx])*r*sv[:, idx])
# # end
# # spec_1 = spec;
#     return (spec, spec.copy())

@njit
def my_correlation_use(stear_vects, iqs) -> np.ndarray:
    """estimate distances using correlation

    Args:
        stear_vects (np.array([num_freqs, num_dists])): ideal normalized IQ values for fixed distances
        iqs (np.array([num_freqs, num_freqs])): IQs * ctranspose(IQs)
    """
    #  TODO: Ask guys to explain (or at least name it) <10-01-22, astadnik> #
    # orig = np.abs(
    #     np.apply_along_axis(
    #         lambda vects: vects @ iqs @ vects.conj().T,
    #         0,
    #         stear_vects,
    #     )
    # )
    # my = np.abs(((stear_vects.T @ iqs) * stear_vects.conj().T).sum(1))
    # assert np.allclose(my, orig)

    return np.abs(((stear_vects.T.astype(np.complex128) @ iqs.astype(np.complex128)) * stear_vects.conj().T).sum(1))
