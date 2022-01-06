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


def my_correlation_use(stear_vects, iqs) -> np.ndarray:
    """estimate distances using correlation

    Args:
        stear_vects (np.array([num_freqs, num_dists])): ideal normalized IQ values for fixed distances
        iqs (np.array([num_freqs, num_freqs])): IQs * ctranspose(IQs)
    """
    # orig = np.abs(
    #     np.apply_along_axis(
    #         lambda vects: vects @ iqs @ vects.conj().T,
    #         0,
    #         stear_vects,
    #     )
    # )

    # my = []
    # for i in range(stear_vects.shape[1]):
    #     vects = stear_vects[:, i]
    #     my.append(vects @ iqs @ vects.conj().T)
    # my = np.abs(my).T

    # my = np.abs(stear_vects.T @ iqs @ stear_vects.conj()).sum(0)
    # ic(my, orig)
    # # print(my)
    # # print(orig)

    # assert (my == orig).all()

    return np.abs(
        np.apply_along_axis(
            lambda vects: vects @ iqs @ vects.conj().T,
            0,
            stear_vects,
        )
    )
