function [score] = dr_plarge(X)
n = size(X,1);
score = zeros(n, 4900);
for i = 1:n
    cur_img = reshape(X(i,:), [100 100 3]);
    I = rgb2gray(uint8(cur_img));
    index = [1:2:30 31:70 71:2:100];
    I = I(index, :);
    I = I(:,index);
    score(i, :) = I(:);
end