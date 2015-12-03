function [score_train, score_test, numpc] = pca_getpc(train_x, test_x)

%   input: original X for training and testing
%   output: PCAed X for training and testing, number of PCs that you
%   selected
[coeff, score, latent] = pca([train_x; test_x]);
score_train = score(1:size(train_x,1), :);
score_test = score(size(train_x,1)+1:end, :);
numpc = find(cumsum(latent)/sum(latent) >= 0.95, 1);
