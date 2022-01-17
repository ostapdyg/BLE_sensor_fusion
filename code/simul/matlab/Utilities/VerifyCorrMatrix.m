function VerifyCorrMatrix(r)

try chol(r);
    % disp('Matrix is symmetric positive definite.')
catch ME
    disp('For a positive semi-definite matrix, the eigenvalues should be non-negative:');
    d = eig(r);
    disp(d);
    disp('ERROR!!!');
    disp('Matrix is not symmetric positive definite')
    
end

end
