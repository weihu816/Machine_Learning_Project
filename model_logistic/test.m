%% import data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];
X_test = [X_word_test X_img_test];

%% PCA
addpath('../lib/pca');
[score_train1, score_test1, numpc1] = pca_getpc(X_train, X_test);
X_train = score_train1(:, 1:numpc1);
X_test = score_test1(:, 1:numpc1);

%% normalization
X_train = atan(X_train) * 2 / pi;
X_test = atan(X_test) * 2 / pi;

%% 
addpath('liblinear');
S = 2;
cost = 0.0017;
% 10.^(-5:1:1)
% P = [0.0001,0.001,0.01,0.1,1];
% E = [0.00001,0.0001,0.001,0.01,0.1,1];
X_train = sparse(X_train);
for s = S
   for c = cost
       train(Y_train, X_train, sprintf('-q -s %g -c %g -v 10', s, c));
       model = train(Y_train, X_train, sprintf('-q -s %g -c %g', s, c));
       fprintf('(%g,%g) -> %g\n', s, c, mean(predict(Y_train, X_train, model, '-q') == Y_train));
   end
end