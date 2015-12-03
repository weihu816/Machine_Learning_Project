%%
% addpath('../lib/libsvm');
% k_poly_linear = @(x,x2) kernel_poly(x, x2, 1);
% k_poly_quadratic = @(x,x2) kernel_poly(x, x2, 2);
% k_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
% % k_intersection = @(x,x2) kernel_intersection(x, x2);
% 
% X = X_train;
% Y = Y_train;
% Xtest = X_train;
% Ytest = Y_train;
% [test_err1 info1] = kernel_libsvm(X, Y, Xtest, Ytest, k_poly_linear);
% test_err1
% [test_err2 info2] = kernel_libsvm(X, Y, Xtest, Ytest, k_poly_quadratic);
% test_err2
% [test_err3 info3] = kernel_libsvm(X, Y, Xtest, Ytest, k_gaussian);
% test_err3
% results.intersect = kernel_libsvm(X, Y, Xtest, Ytest, k_intersection);

%%
addpath('libsvm');
k_intersection = @(x,x2) kernel_intersection(x, x2);

K = k_intersection(X_train, X_train);
Ktest = k_intersection(X_train, X_test);

% Use built-in libsvm cross validation to choose the C regularization
% parameter.
crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K,1))' K], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);

fprintf('Cross-val chose best C = %g\n', crange(bestc));
% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y_train, [(1:size(K,1))' K], sprintf('-q -t 4 -c %g', 0.0008));
Ytest = ones(size(X_test,1), 1);
[yhat, acc, vals] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);

dlmwrite('submit.txt', yhat);

% k_poly_quadratic
% ************************************************************************
% Your project results as of Sat Nov 21 20:53:42 2015:
% ************************************************************************
% Team: Top Learner
% Accuracy: 0.8367

% k_poly_linear
% ************************************************************************
% Your project results as of Sun Nov 22 09:06:01 2015:
% ************************************************************************
% Team: Top Learner
% Accuracy: 0.8627

% k_intersection
% ************************************************************************
% Your project results as of Sun Nov 22 14:09:04 2015:
% ************************************************************************
% Team: Top Learner
% Accuracy: 0.8827
