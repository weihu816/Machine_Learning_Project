%% start
tic
addpath('./lib/pca');
addpath('./lib/liblinear');

[score_train1, score_test1, numpc1] = pca_getpc(X_word_train, X_word_test);
[score_train2, score_test2, numpc2] = pca_getpc(X_img_train, X_img_test);

X_train = score_train1(:, 1:numpc1);
X_test = score_test1(:, 1:numpc1);

model = train(Y_train, sparse([X_img_train X_word_train]), ['-s 7', 'col']);

[predictions] = predict(Y_train, sparse([X_img_train X_word_train]), model, ['-q', 'col']);
precision_train = 1 - sum(predictions~=Y_train) / length(Y_train);

[predicted_label_test] = predict(Y_test, sparse([X_img_test X_word_test]), model, ['-q', 'col']);
precision_test = 1 - sum(predicted_label_test~=Y_test) / length(Y_test)
toc
'end2'
%% For submission
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

addpath('./lib/pca');
addpath('./lib/liblinear');

model = train(Y_train, sparse([X_img_train X_word_train]), '-s 0 -v 10 -q');

[predictions] = predict(Y_train, sparse([X_img_train X_word_train]), model, ['-q', 'col']);
precision_train = 1 - sum(predictions~=Y_train) / length(Y_train);

B_first = mnrfit(sparse([X_img_train X_word_train]), Y_train + 1);
predicted_label_train2 = 1 - predict(B_first, [ones(size(X_word,1),1) X_word]);
precision_train2 = sum(prediction_first == Y_train) / length(Y_train);

[predicted_label_test] = predict(ones(size([X_img_test X_word_test], 1),1), sparse([X_img_test X_word_test]), model, ['-q', 'col']);

% Use turnin on the output file
% turnin -c cis520 -p leaderboard submit.txt
dlmwrite('submit.txt', predicted_label_test);
%%
addpath('./lib/pca');
[score_train1, score_test1, numpc1] = pca_getpc(X_train, X_test);
X_train = score_train1(:, 1:numpc1);
X_test = score_test1(:, 1:numpc1);


%% test of logistic regression
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
%  train(Y_train, X_train, '-q -s 0 -c 0.86 -v 10')