X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];
X_test = [X_word_test X_img_test];

%normalization 1
X_train = atan(X_train)*2/pi;

%construct the classifier
mdl = fitcknn(X_train, Y_train);

%change neighbor size (default is 1): 
mdl.NumNeighbors = 16;

%make predictions
predictions = predict(mdl, X_test);

%generate submit.txt
dlmwrite('submit.txt', predictions);
