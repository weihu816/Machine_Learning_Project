addpath('./lib/libsvm');
k_poly_linear = @(x,x2) kernel_poly(x, x2, 1);
k_poly_quadratic = @(x,x2) kernel_poly(x, x2, 2);
k_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
k_intersection = @(x,x2) kernel_intersection(x, x2);

K1 = k_poly_linear(X_train, X_train);
K2 = k_poly_quadratic(X_train, X_train);
K3 = k_gaussian(X_train, X_train);
K4 = k_intersection(X_train, X_train);

crange = 10.^[-10:2:3];

for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K1,1))' K1], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));
acc(bestc)

for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K2,1))' K2], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));
acc(bestc)

for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K3,1))' K3], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));
acc(bestc)

for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K4,1))' K4], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));
acc(bestc)

