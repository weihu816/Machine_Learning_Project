X_img_train = importdata('train/image_features_train.txt');
X_images_train = importdata('train/images_train.txt');
X_word_train = importdata('train/words_train.txt');
Y_train = importdata('train/genders_train.txt');

save('train/train.mat', 'X_img_train', 'X_images_train', 'X_word_train', 'Y_train');

X_img_test = importdata('test/image_features_test.txt');
X_images_test = importdata('test/images_test.txt');
X_word_test = importdata('test/words_test.txt');

save('test/test.mat', 'X_img_test', 'X_images_test', 'X_word_test');
