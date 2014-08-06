function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
%theta = reshape(theta, numClasses, inputSize+1);
%theta = [theta(:,end) theta(:,1:inputSize)];
%data = [ones(1, size(data,2)); data];

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
%labelid = 1:numClasses;					%1*M
%groundTruth = full(sparse(labelid,labels,1));		%

M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);
thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;

%display(M(1:5,1:5));
%display(data(1:5,1:5));
%display(size(theta));
%display(theta(1:5,end));
%display(theta(1:5,1:5));
%display((theta*data)(1:5,1:5));
%display(p(1:5,1:5));
%display(thetagrad(1:5,1:5));


% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

