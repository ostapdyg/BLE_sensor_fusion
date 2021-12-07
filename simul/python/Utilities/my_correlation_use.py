import numpy as np

def ctranspose(arr: np.ndarray) -> np.ndarray:
    # Explanation of the math involved:
    # x      == Real(X) + j*Imag(X)
    # conj_x == Real(X) - j*Imag(X)
    # conj_x == Real(X) + j*Imag(X) - 2j*Imag(X) == x - 2j*Imag(X)
    tmp = arr.transpose()
    return tmp - 2j*tmp.imag

# function [spec, spec_1] = my_correlation_use(sv, signal, r)
def my_correlation_use(sv, signal, r)->tuple[np.array, np.array]:
# % spec = abs(sv' * signal).^2;

# spec = zeros(size(sv, 2), 1);
    spec = np.zeros((np.size(sv, 1), 1))
# for idx = 1:size(sv, 2)
    for idx in range(np.size(sv, 1)):
#     spec(idx) = abs(sv(:, idx)' * r * sv(:, idx));
        spec[idx] = np.abs(ctranspose(sv[:,idx])*r*sv[:, idx])
# end
# spec_1 = spec;
    return (spec, spec.copy())


if(__name__ == "__main__"):
    A = np.array([[1,-2j],[2j,5]])
    print(my_correlation_use(A, 10, 10))