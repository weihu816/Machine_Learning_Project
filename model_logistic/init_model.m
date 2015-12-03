function model = init_model()

addpath('./lib/liblinear');
load('model_logistic.mat');
model = model_logistic;

% X_img_train = importdata('../train/image_features_train.txt');
% X_word_train = importdata('../train/words_train.txt');
% Y_train = importdata('../train/genders_train.txt');
% model_logistic = train(Y_train, sparse([X_img_train X_word_train]), '-s 2 -c 0.5 -q');