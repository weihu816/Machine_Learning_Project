function predictions = make_final_prediction(model,X_test,X_train)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% we don't need image
X_test = X_test(:,[1:5000 35001:35007]);
X_test = X_test(:,model.index);
P1 = predict(model.model1, full(X_test));

X_test = sparse(X_test);
addpath('libsvm');
k_intersection = @(x,x2) kernel_intersection(x, x2);
Ktest = k_intersection(model.X_train_svm, X_test);
[P2, ~, ~] = svmpredict(ones(size(X_test,1), 1), [(1:size(Ktest,1))' Ktest], model.model2);

addpath('liblinear');
X_test(:,1:end-7) = atan(X_test(:,1:end-7)) * 2 / pi; % X_test is changed here
P3 = predict(ones(size(X_test,1),1), sparse(X_test), model.model3, '-q');
predictions = (P1 + P2 + P3) ./ 3;
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;

