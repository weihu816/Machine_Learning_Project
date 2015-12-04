%% load data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];

%normalization 1
X_train = atan(X_train)*2/pi;

%% construct the knn classifier
mdl = fitcknn(X_train, Y_train);

%finding suitable k 
for i=1:1:20
    %change neighbor size (default is 1): 
    mdl.NumNeighbors = i;

    %resubstitution error
    rloss = resubLoss(mdl);

    %cross validation error
    cvmdl = crossval(mdl, 'kfold', 10);
    kloss = kfoldLoss(cvmdl);
    fprintf('k = %i\n resubstitution accuracy: %.5f%%, cross validation accuracy: %.5f%%\n\n', i, (1-rloss)*100, (1-kloss)*100);
end