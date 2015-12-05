%% load data
X_img_train = importdata('../train/image_features_train.txt');
X_word_train = importdata('../train/words_train.txt');
Y_train = importdata('../train/genders_train.txt');

X_img_test = importdata('../test/image_features_test.txt');
X_word_test = importdata('../test/words_test.txt');

X_train = [X_word_train X_img_train];

%normalization 1
X_train = atan(X_train)*2/pi;

%%
model = fitensemble(X_train, Y_train, 'AdaBoostM1', 2000, 'Tree');
prediction = predict(model, X_test);
dlmwrite('submit.txt', prediction);

%%
count = 2000;
m1 = fitensemble(X_train, Y_train, 'AdaBoostM1', count, 'Tree');
m2 = fitensemble(X_train, Y_train, 'LogitBoost', count, 'Tree');
m3 = fitensemble(X_train, Y_train, 'GentleBoost', count, 'Tree');
m4 = fitensemble(X_train, Y_train, 'RobustBoost', count, 'Tree');
m5 = fitensemble(X_train, Y_train, 'LPBoost', count, 'Tree');
m6 = fitensemble(X_train, Y_train, 'TotalBoost', count, 'Tree');
m7 = fitensemble(X_train, Y_train, 'RUSBoost', count, 'Tree');

p = zeros(size(X_test, 1), 7);
p(:, 1) = predict(m1, X_test);
p(:, 2) = predict(m2, X_test);
p(:, 3) = predict(m3, X_test);
p(:, 4) = predict(m4, X_test);
p(:, 5) = predict(m5, X_test);
p(:, 6) = predict(m6, X_test);
p(:, 7) = predict(m7, X_test);

submit = zeros(size(X_test, 1), 1);
submit(mean(p, 2) > 0.5) = 1;
dlmwrite('boosting_average.txt', submit);
