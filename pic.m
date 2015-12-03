load('train/train.mat');
load('test/test.mat');
%% male
avg_image = mean(X_images_train(Y_train == 0, :), 1);
cur_img = reshape(avg_image, [100 100 3]);
imshow(uint8(cur_img));
%% female
avg_image = mean(X_images_train(Y_train == 1, :), 1);
cur_img = reshape(avg_image, [100 100 3]);
imshow(uint8(cur_img));