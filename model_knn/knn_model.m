%% import data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];
X_test = [X_word_test X_img_test];

%normalization
X_train = atan(X_train)*2/pi;
X_test = atan(X_test)*2/pi;

%% construct the classifier
mdl = fitcknn(X_train, Y_train);

%change neighbor size to 16 (which has the best cross validation accuracy)
mdl.NumNeighbors = 16;

%make predictions on test set
predictions = predict(mdl, X_test);

%generate submit.txt
dlmwrite('submit.txt', predictions);
