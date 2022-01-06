import numpy as np


def ctranspose(arr: np.ndarray) -> np.ndarray:
    # Explanation of the math involved:
    # x      == Real(X) + j*Imag(X)
    # conj_x == Real(X) - j*Imag(X)
    # conj_x == Real(X) + j*Imag(X) - 2j*Imag(X) == x - 2j*Imag(X)
    tmp = arr.transpose()
    return tmp - 2j*tmp.imag

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
    res = np.zeros(np.shape(stear_vects)[1])
    # for dist_idx in range(np.size(res)):
    #     corr = np.abs(
    #         np.array([stear_vects[:, dist_idx]]).dot(
    #             iqs).dot(
    #             ctranspose(np.array([stear_vects[:, dist_idx]])))
    #     )
    #     # print(f"   my_correlation_use iqs:{iqs.shape}")
    #     # print(f"   my_correlation_use corr:{corr.shape}")
    #     res[dist_idx] = corr
    # return res
    return np.apply_along_axis(lambda vects: np.abs(
            np.array([vects]).dot(
                iqs).dot(
                ctranspose(np.array([vects])))),
                0, stear_vects)
# 

if(__name__ == "__main__"):
    pass
    # A = np.array([[1,-2j],[2j,5]])
    # print(my_correlation_use(A, 10, 10))
