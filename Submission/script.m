X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');
X_train = [X_word_train X_img_train];
[~, ~, sigma] = zscore(X_train);
X_train = X_train(:, sigma ~= 0);
index = randperm(size(X_train, 1))';
index_train = index(1:4200);
X_train = sparse(X_train(index_train, :));
Y_train = Y_train(index_train,:);
addpath('libsvm');
k_intersection = @(x,x2) kernel_intersection(x, x2);
K = k_intersection(X_train, X_train);
model_svm = svmtrain(Y_train, [(1:size(K,1))' K], sprintf('-q -t 4 -c 0.0005'));
save('model_svm.mat', 'model_svm');
save('X_train.mat', 'X_train');


X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');
X_train = X_word_train;
[~, ~, sigma] = zscore(X_train);
X_train = X_train(:, sigma ~= 0);
addpath('liblinear');
X_train = atan(X_train) * 2 / pi;
X_train = [X_train X_img_train];
X_train = sparse(X_train);
model_logistic = train(Y_train, X_train, sprintf('-q -s 2 -c 0.00161'));
save('model_logistic.mat', 'model_logistic');


X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');
X_train = [X_word_train X_img_train];
[~, ~, sigma] = zscore(X_train);
X_train = X_train(:, sigma ~= 0);
model_em = fitensemble(X_train, Y_train, 'AdaBoostM1', 1200, 'Tree');
save('model_em.mat', 'model_em');







X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X = [X_word_train X_img_train];
Y = Y_train;

clear X_img_train
clear X_word_train
clear Y_train

[~, ~, sigma] = zscore(X);

[sortedValues,sortIndex] = sort(sigma,'descend'); 

X = X(:, sortIndex(1:3000));

index = randperm(size(X, 1))';
index_train = index(1:3000);
index_test = index(3001:end);

X_train = X(index_train, :);
X_test = X(index_test, :);
Y_train = Y(index_train);
Y_test  = Y(index_test);

clear index
clear index_train
clear index_test


model1 = fitensemble(X_train, Y_train, 'AdaBoostM1', 1500, 'Tree');
model2 = fitensemble(X_train, Y_train, 'AdaBoostM1', 1200, 'Tree');
mean(predict(model1, X_train)==Y_train)
mean(predict(model1, X_test)==Y_test)
mean(predict(model2, X_train)==Y_train)
mean(predict(model2, X_test)==Y_test)
