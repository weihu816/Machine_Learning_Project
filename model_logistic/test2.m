%% import data
X_img_train = importdata('./train/image_features_train.txt');
X_word_train = importdata('./train/words_train.txt');
Y_train = importdata('./train/genders_train.txt');

% X_img_test = importdata('test/image_features_test.txt');
% X_word_test = importdata('test/words_test.txt');

X_img_test = X_img_train(3001:end, :);
X_word_test = X_word_train(3001:end, :);
Y_test = Y_train(3001:end, :);

X_img_train = X_img_train(1:3000, :);
X_word_train = X_word_train(1:3000, :);
Y_train = Y_train(1:3000, :);
'end1'
%% try different cost of logistic regression

addpath('./lib/pca');
addpath('./lib/liblinear');

[score_train1, score_test1, numpc1] = pca_getpc(X_word_train, X_word_test);
[score_train2, score_test2, numpc2] = pca_getpc(X_img_train, X_img_test);

X_train = score_train1(:, 1:numpc1);
X_test = score_test1(:, 1:numpc1);

%% test of logistic regression
model_type = [0 6 7]
cost = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]
percision_test = zeros(10,1);
i=1;
for cost = [1 2 3 4 5]
    para1 = ['-s ', num2str(2)];
    para2 = ['-c ', num2str(cost)]
    model = train(Y_train, sparse([X_img_train X_word_train]), [para1,'-v 10', 'col']);
    [predicted_label_train] = predict(Y_train, sparse([X_img_train X_word_train]), model, ['-q', 'col']);
    precision_train = 1 - sum(predicted_label_train~=Y_train) / length(Y_train);

    [predicted_label_test] = predict(Y_test, sparse([X_img_test X_word_test]), model, ['-q', 'col']);
    percision_test(i) = 1 - sum(predicted_label_test~=Y_test) / length(Y_test)
    i = i+1;
end

'end2'
