function [spec, spec_1] = my_correlation_use(sv, signal, r)

% spec = abs(sv' * signal).^2;
spec = zeros(size(sv, 2), 1);
for idx = 1:size(sv, 2)
    spec(idx) = abs(sv(:, idx)' * r * sv(:, idx));
end
spec_1 = spec;
