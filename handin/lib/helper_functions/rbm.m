function [ dbn ] = rbm( train_x )
    
    %  ex2 train a DBN. Its weights can be used to initialize a NN.
    rand('state',0)
    %train dbn
    dbn.sizes = [1000];
    opts.numepochs =   12;
    opts.batchsize = 98;
    opts.momentum  =   0;
    opts.alpha     =   1;
    dbn = dbnsetup(dbn, train_x, opts);
    dbn = dbntrain(dbn, train_x, opts);
    figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

end

