load('../train/train.mat');
load('../test/test.mat');

X_train = [X_img_train X_word_train; X_img_test(xxx == yyy,:) X_word_test(xxx == yyy, :)];
X_test = [X_img_test X_word_test];

%% pca
addpath('../lib/pca');

% normalize
[X_train, mu, sigma] = zscore(X_train);
X_test = normalize(X_test, mu, sigma);

[score_train, score_test, numpc] = pca_getpc( X_train, X_test );
% [score_train, score_test, numpc] = pca_getpc( X_img_orig_train, X_img_orig_test );


%% auto encoder
addpath('../lib/DL_toolbox/util','../lib/DL_toolbox/NN','../lib/DL_toolbox/DBN');

% X_mean = mean(X_train, 1);
% X_std = std(X_train, 1);
% train_x = bsxfun(@minus, X_train, X_mean);
% train_x = train_x ./ repmat(X_std, size(X_train, 1), 1);
% test_x = bsxfun(@minus, X_test, X_mean);
% test_x = test_x ./ repmat(X_std, size(X_test, 1), 1);
x_mean = mean(X_train, 1);
train_x = bsxfun(@minus, X_train, x_mean);
x_min = min(train_x, [], 1);
train_x = bsxfun(@plus, train_x, -x_min);
x_max = max(train_x, [], 1);
x_max(x_max == 0) = 1;
train_x = train_x ./ repmat(x_max, size(train_x, 1), 1);


[ dbn ] = rbm( train_x );
[ new_feat, new_feat_test ] = newFeature_rbm( dbn,train_x,train_x );


%% logistic 
addpath('../lib/liblinear');

[ precision_ori_log ] = logistic( X_train, Y_train, X_test, Y_test );
[ precision_pca_log ] = logistic( score_train(:, 1:numpc), Y_train, score_train(:,1:numpc), Y_train );
% [ precision_ae_log ] = logistic( new_feat, Y_train, new_feat_test, Y_test );

model = train(Y_train, sparse(X_train), ['-s 0', 'col']);
[predicted_label] = predict(Y_test, sparse(X_test), model, ['-q', 'col']);
precision = sum(predicted_label==Y_train) / length(Y_train);

[prediction] = predict(ones(size(X_test,1),1), sparse(score_test(:,1:numpc)), model, ['-q', 'col']);
dlmwrite('submit.txt', prediction);

%% kmeans

% K = [10, 25];
% precision_ae_km = zeros(length(K), 1);
% for i = 1 : length(K)
%     k = K(i);
%     precision_ae_km(i) = k_means(new_feat, Y_train, new_feat_test, Y_test, k);
% end
% 
% 
% precision_pc_km = zeros(length(K), 1);
% for i = 1 : length(K)
%     k = K(i);
%     precision_pc_km(i) = k_means(score_train(:, 1:numpc), Y_train, score_test(:,1:numpc), Y_test, k);
% end
