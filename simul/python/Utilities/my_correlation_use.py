import numpy as np

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


def my_correlation_use(stear_vects, iqs) -> np.ndarray:
    """estimate distances using correlation

    Args:
        stear_vects (np.array([num_freqs, num_dists])): ideal normalized IQ values for fixed distances
        iqs (np.array([num_freqs, num_freqs])): IQs * ctranspose(IQs)
    """
    #  TODO: WTF :) <06-01-22, astadnik> #
    return np.apply_along_axis(
        lambda vects: np.abs(
            np.array([vects]).dot(iqs).dot(np.array([vects]).conj().T)
        ),
        0,
        stear_vects,
    )
