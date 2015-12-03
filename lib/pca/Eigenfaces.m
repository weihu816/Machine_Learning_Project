% Matrix faces has dimensions faceheight x facewidth x numfaces.
% npca = # of features after PCA
function [Xpca, V, mu] = Eigenfaces(faces, npca)

% Make each face example a row in a matrix
numrows = size(faces, 1);
numcols = size(faces, 2);
numfaces = size(faces, 3);
X0 = reshape(faces, prod([numrows, numcols]), numfaces);
X0 = X0';

% Subtract the mean value of each feature
mu = mean(X0,1);
X = bsxfun(@minus, X0, mu);

% Get the eigenvectors
V = pca(X, npca);

% Plot the eigenvectors as eigenfaces
colormap gray
for i = 1:npca
    imagesc(reshape(V(:,i), numrows, numcols))
    pause
end

% Project the faces onto the eigenvectors
% (multiply each face by each eigenvector)
Xpca = X * V;

% Plot the original image followed by the image reconstructed from the PCA
for i = 1:numfaces
    % Plot the original image
    imagesc(faces(:,:,i));
    pause
    
    % Take the product of each eigenvector with the weight this face places
    % on that eigenvector, summed over all eigenvectors
    % (sum the contributions of all our new feature dimensions)
    imagesc(reshape(sum(diag(Xpca(i,:))*V',1), numrows, numcols));
    pause
end

return;

% X should have n samples with m features in an nxm matrix.
% This method returns eigenvectors as columns of V.
function V = pca(X, npca)

% Compute the variances matrix (mxm)
sigma = X'*X;

% D = diagonal with npca largest eigenvalues,
% V = corresponding eigenvectors
% ('lm' for 'largest magnitude' eigenvalues)
[V,D] = eigs(sigma, npca, 'lm');

% You could alternatively call the SVD function here instead of eigs,
% if you want to try the SVD implementation of PCA.

return;