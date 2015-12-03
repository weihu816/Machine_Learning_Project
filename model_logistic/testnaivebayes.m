%%
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

% X_img_test = importdata('test/image_features_test.txt');
% X_word_test = importdata('test/words_test.txt');

X_img_test = X_img_train(3001:end, :);
X_word_test = X_word_train(3001:end, :);
Y_test = Y_train(3001:end, :);

X_img_train = X_img_train(1:3000, :);
X_word_train = X_word_train(1:3000, :);
Y_train = Y_train(1:3000, :);
'end1'

%% naive bayes
tic
load fisheriris
[score_train, score_test, numpc] = pca_getpc_mean([X_img_train X_word_train], [X_img_test X_word_test]);
X_train = score_train(:, 1:numpc);
X_test = score_test(:,1:numpc);
nb = fitNaiveBayes(X_train, Y_train);

[predicted_label_train] = nb.predict(X_train);
precision_train = 1 - sum(predicted_label_train~=Y_train) / length(Y_train)
[predicted_label_test] = nb.predict(X_test );
precision_test = 1 - sum(predicted_label_test~=Y_test) / length(Y_test)
toc
