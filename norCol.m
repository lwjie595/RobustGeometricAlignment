function Y=norCol(X)
% % Normalize each column vector in matrix X

% 
Max=max(X,[],2);
Min=min(X,[],2);
Y=bsxfun(@rdivide,bsxfun(@minus,X,Min),Max-Min);
