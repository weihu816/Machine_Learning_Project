function demo_NN
addpath('../lib/DL_toolbox/util','../lib/DL_toolbox/NN','../lib/DL_toolbox/CNN','../lib/DL_toolbox/SAE');

load('../train/train.mat');
load('../test/test.mat');

X_train = sparse([X_img_train X_word_train]);
% X_test = sparse([X_img_test X_word_test]);

index = randperm(size(X_train, 1));
index1 = index(1:3000);
index2 = index(3001:end);
train_x = sparse(X_train(index1, :));
test_x = sparse(X_train(index2, :));
train_y = [zeros(length(Y_train(index1)), 1) Y_train(index1)];
test_y  = [zeros(length(Y_train(index2)), 1) Y_train(index2)];

% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);

% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
% [784 = input dimension, 100 = size of hidden layer, 10 = size of ...
%   output]  
% to add more hidden layers, make the vector longer.
nn = nnsetup([5007 100 2]);  

opts.numepochs =  25;   %  Number of full sweeps through data
opts.batchsize = 100;  %  Take a mean gradient step over this many samples
[nn, loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
disp('Testing error for vanilla neural network: ');
disp(er);

%% ex2 neural net with L2 weight decay
rand('state',0)

% [784 = input dimension, 
% 100 = size of hidden layer, 
% 10 = size of output] 
nn = nnsetup([5007 100 2]); 
nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  25;       %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

[nn loss] = nntrain(nn, train_x, train_y, opts);

[er, bad] = nntest(nn, test_x, test_y);
disp('Testing error for neural network with L2 weight decay: ');
disp(er);


