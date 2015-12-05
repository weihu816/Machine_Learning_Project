% %%
% X_img_train = importdata('../train/image_features_train.txt');
% X_word_train = importdata('../train/words_train.txt');
% Y_train = importdata('../train/genders_train.txt');
% 
% X_img_test = importdata('../test/image_features_test.txt');
% X_word_test = importdata('../test/words_test.txt');
% 
% p1 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 10)
% p2 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 20)
% p3 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 50)
% p4 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 100)
% p5 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 200)
% p6 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 500)
% p7 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 1000)
% p8 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 2000)
% p9 = k_means([X_img_train X_word_train], Y_train, [X_img_train X_word_train], Y_train, 4000)


X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

[~, ~, sigma] = zscore(X_word_train);
sigma = sigma';
K=4000;
label = zeros(K,1);
[IDX,C] = kmeans(sigma, K, 'MaxIter', 1);
