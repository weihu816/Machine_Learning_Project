function [ new_feat,new_feat_test ] = newFeature_rbm( dbn, train_x, test_x )

    
    % unfold dbn to nn
    nn = dbnunfoldtonn(dbn, 784);
    nn.activation_function = 'sigm';

    % Get new training data features
    nn2 = nnff(nn, train_x, train_x);
    new_feat=nn2.a{2};
    
    % Get new test data features
    nn3 = nnff(nn, test_x, test_x);
    new_feat_test = nn3.a{2};

end

