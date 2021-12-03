function [spec,spec_1] = MYmusicdoa_eigen_use(Nsig, sv, eigenvects)
%musicdoa   MUSIC direction of arrival (DOA)

% Separate the signal and noise eigenvectors
noise_eigenvects = eigenvects(:,Nsig+1:end);

% Calculate the spatial spectrum. Add a small positive constant to prevent
% division by zero.
D = sum(abs((sv'*noise_eigenvects)).^2,2)+eps(1);
spec_1 = D;
spec = sqrt(1./D).';

end
