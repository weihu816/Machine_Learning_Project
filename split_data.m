%% Save data
X_img_train = importdata('train/image_features_train.txt');
X_images_train = importdata('train/images_train.txt');
X_word_train = importdata('train/words_train.txt');
Y_train = importdata('train/genders_train.txt');

X_img_test = importdata('test/image_features_test.txt');
X_images_test = importdata('test/images_test.txt');
X_word_test = importdata('test/words_test.txt');
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
%%
%% 
load('train/train.mat');
load('test/test.mat');

X_train = sparse([X_img_train X_word_train]);
X_test = sparse([X_img_test X_word_test]);

clear X_images_train
clear X_img_train
clear X_word_train
clear X_images_test
clear X_img_test
clear X_word_test
%% filter uninformative features
%% 
load('train/train.mat');
load('test/test.mat');

X_train = sparse([X_img_train X_word_train]);
X_test = sparse([X_img_test X_word_test]);
[~, ~, sigma] = zscore(X_train);
X_train = X_train(:, sigma ~= 0);
X_test = X_test(:, sigma ~= 0);
X_train = atan(X_train) * 2 / pi;
X_test = atan(X_test) * 2 / pi;
clear X_images_train
clear X_img_train
clear X_word_train
clear X_images_test
clear X_img_test
clear X_word_test
clear sigma
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