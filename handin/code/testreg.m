%% load data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');
%% reg
%do normalization of the data
X_img_train = atan(X_img_train)*2/pi;
X_word_train = atan(X_word_train)*2/pi;
X_img_test = atan(X_img_test)*2/pi;
X_word_test = atan(X_word_test)*2/pi;
 
addpath('../lib/liblinear');

% using cross validation we have found that c = 0.0012 is best
% count=1;
% for i = [0.085,0.009,0.0095,0.0098,0.01,0.012,0.014,0.016,0.018,0.11,0.12,0.13]
%     param = ['-s 0 -v 10 -c ' num2str(i)]
%     acc(count) = train(Y_train, sparse([X_img_train X_word_train]), param);
%     count = count+1
% end

param = ['-s 0 -c 0.01'];
model = train(Y_train, sparse([X_img_train X_word_train]), param);

% get the training error
[predicted_label_train] = predict(Y_train, sparse([X_img_train X_word_train]), model, ['-q', 'col']);
precision_train = 1 - sum(predicted_label_train~=Y_train) / length(Y_train)

% get the test result file
[predicted_label_test] = predict(ones(size([X_img_test X_word_test], 1),1), sparse([X_img_test X_word_test]), model, ['-q', 'col']);
dlmwrite('submit.txt', predicted_label_test)