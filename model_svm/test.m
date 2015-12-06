%% import data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];
X_test = [X_word_test X_img_test];

%% pre-computed kernels
addpath('libsvm');
k_poly_linear = @(x,x2) kernel_poly(x, x2, 1);
k_poly_quadratic = @(x,x2) kernel_poly(x, x2, 2);
k_gaussian = @(x,x2) kernel_gaussian(x, x2, 20);
k_intersection = @(x,x2) kernel_intersection(x, x2);

K1 = k_poly_linear(X_train, X_train);
K2 = k_poly_quadratic(X_train, X_train);
K3 = k_gaussian(X_train, X_train);
K4 = k_intersection(X_train, X_train);

K1_test = k_poly_linear(X_train, X_test);
K2_test = k_poly_quadratic(X_train, X_test);
K3_test = k_gaussian(X_train, X_test);
K4_test = k_intersection(X_train, X_test);

%%
% set the kernel
K = K4;
Ktest = K4_test;

% Use libsvm cross validation to choose the C regularization parameter
crange = 0.0001:0.0001:1;
for i = 1:numel(crange)
    acc(i) = svmtrain(Y_train, [(1:size(K,1))' K], sprintf('-q -t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

% Train and evaluate SVM classifier using libsvm
model = svmtrain(Y_train, [(1:size(K,1))' K], sprintf('-q -t 4 -c %g', c));
[yhat, acc, vals] = svmpredict(ones(size(X_test,1), 1), [(1:size(Ktest,1))' Ktest], model);

dlmwrite('submit.txt', yhat);

% best parameter is 0.0008 with histgram intersection kernel