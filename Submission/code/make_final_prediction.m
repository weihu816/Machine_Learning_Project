function predictions = make_final_prediction(model,X_test)
% Input
% X_test : a nxp vector representing "n" test samples with p features.
% X_test=[words images image_features] a n-by-35007 vector
% model : struct, what you initialized from init_model.m
%
% Output
% prediction : a nx1 which is your prediction of the test samples

% we don't need image
X_test = X_test(:,[35001:35007 1:5000]);
X_test = X_test(:,model.index);
P1 = predict(model.model1, full(X_test));
addpath('liblinear');

% normolize
X_test = atan(X_test) * 2 / pi;
P2 = predict(ones(size(X_test,1),1), sparse(X_test), model.model2, '-q');
predictions = (P1 + P2) ./ 2;
predictions(predictions > 0.5) = 1;
predictions(predictions <= 0.5) = 0;

