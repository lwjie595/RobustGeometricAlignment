function distM=distance(X,Y)
% Computing the squared Euclidean distance matrix
D=size(X,1);
X2=sum(X.^2,1);
if nargin>=2
    d=size(Y,1);
    if D~=d
        error('Both sets of vectors must have same dimensionality!\n');
    end
    Y2=sum(Y.^2,1);
    distM=bsxfun(@plus,X2',bsxfun(@plus,Y2,-2*X'*Y));

else
    distM=bsxfun(@plus,X2',bsxfun(@plus,X2,-2*X'*X));
end

