%% load data
addpath('../lib/helper_functions');
addpath('../lib/DL_toolbox/util','../lib/DL_toolbox/NN','../lib/DL_toolbox/DBN');

X_img_train = importdata('../train/images_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/images_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_img_train = dr_plarge(X_img_train);
X_img_test = dr_plarge(X_img_test);

X_train = X_img_train;
X_test = X_img_test;

%normalization
x_mean = mean(X_train, 1);
train_x = bsxfun(@minus, X_train, x_mean);
x_min = min(train_x, [], 1);
train_x = bsxfun(@plus, train_x, -x_min);
x_max = max(train_x, [], 1);
x_max(x_max == 0) = 1;
train_x = train_x ./ repmat(x_max, size(train_x, 1), 1);

x_mean = mean(X_test, 1);
test_x = bsxfun(@minus, X_test, x_mean);
x_min = min(test_x, [], 1);
test_x = bsxfun(@plus, test_x, -x_min);
x_max = max(test_x, [], 1);
x_max(x_max == 0) = 1;
test_x = test_x ./ repmat(x_max, size(test_x, 1), 1);

%% auto encoder
[ dbn ] = rbm( train_x );
[ new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,train_x );

%% logistic regression
addpath('../lib/liblinear');

% cross validation to find best parameter
count=1;
for i = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    param = ['-s 0 -v 10 -c ' num2str(i)]
    acc(count) = train(Y_train, sparse(new_feat), param);
    count = count+1
end

param = ['-s 0 -c 0.4'];
model = train(Y_train, sparse(new_feat), param);

% get the training error
[predicted_label_train] = predict(Y_train, sparse(new_feat), model, ['-q', 'col']);
precision_train = 1 - sum(predicted_label_train~=Y_train) / length(Y_train);

% get the prediction result file
[predicted_label_test] = predict(ones(size(new_feat_test, 1),1), sparse(new_feat_test), model, ['-q', 'col']);
dlmwrite('submit.txt', predicted_label_test);