%% load data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');
X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

% detele the features column whose column variabce is 0
X_word_train = X_word_train(:,var(X_word_train)~=0);
X_word_test = X_word_test(:,var(X_word_test)~=0);

%% Using naive bayes to fit the model
X_train = [X_img_train X_word_train];
X_test = [X_img_test X_word_test];
nb = fitNaiveBayes(X_train, Y_train);

% get the training error
[predicted_label_train] = nb.predict(X_train);
precision_train = 1 - sum(predicted_label_train~=Y_train) / length(Y_train);

% get the test file
[predicted_label_test] = nb.predict(X_test);
dlmwrite('submit.txt', predicted_label_test);