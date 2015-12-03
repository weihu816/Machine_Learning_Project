%% 3000 training data and 2000 test data (word and image features)
load('train/train.mat');

X = sparse([X_img_train X_word_train]);
Y = Y_train;

index = randperm(size(X, 1))';
index_train = index(1:3000);
index_test = index(3001:end);

X_train = sparse(X(index_train, :));
X_test = sparse(X(index_test, :));
Y_train = Y(index_train);
Y_test  = Y(index_test);

clear index
clear index_train
clear index_test
clear X
clear Y
clear X_images_train
clear X_img_train
clear X_word_train
%% 3000 training data and 2000 test data (original image)

load('train/train.mat');

X = X_images_train;
Y = Y_train;

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
clear X
clear Y
clear X_images_train
clear X_img_train
clear X_word_train