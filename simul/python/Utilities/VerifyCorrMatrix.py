import numpy as np

# function VerifyCorrMatrix(r)
def VerifyCorrMatrix(r):
    try:
        np.linalg.cholesky(r)
        # print('Matrix is symmetric positive definite.')
        # print(np.linalg.eigvals(r))

    except np.linalg.LinAlgError as err:
        print('For a positive semi-definite matrix, the eigenvalues should be non-negative:')
        print(np.linalg.eigvals(r))
        print(r)
        print("ERROR!!!")
        print("Matrix is not symmetric positive definite")
# try chol(r);
#     % disp('Matrix is symmetric positive definite.')
# catch ME
#     disp('For a positive semi-definite matrix, the eigenvalues should be non-negative:');
#     d = eig(r);
#     disp(d);
#     disp('ERROR!!!');
#     disp('Matrix is not symmetric positive definite')
    
# end

# end
if(__name__ == "__main__"):
    A = np.array([[1,-2j],[2j,5]])
    VerifyCorrMatrix(A)