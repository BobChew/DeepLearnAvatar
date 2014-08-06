function [dW1 acc result] = UFLDL_Softmax(train_data,train_labels,test_data,test_labels,func,numiter,numClasses,theta,lambda)

printf('Running Softmax using UFLDL...\n');
%%======================================================================
%% STEP 0: Initialise constants and parameters
%
inputSize = size(train_data,2);

%%======================================================================
%% STEP 1: Load data
%
printf('Normalizing trainging data...\n');
[train_data, mu, sigma] = zscore(train_data);   %  mean & max
test_data = normalize(test_data, mu, sigma);
%train_labels(train_labels==0) = 10; % Remap 0 to 10
%test_labels(test_labels==0) = 10;
train_data = train_data';
test_data = test_data';

%%======================================================================
%% STEP 2: Implement softmaxCost
%
theta = theta(1:inputSize*numClasses);

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, train_data, train_labels);
dW1 = reshape(grad(1:numClasses*inputSize), numClasses, inputSize);
%csvwrite('Softmax_gradient_UF_r1.csv', dW1);                                     

%%======================================================================
%% STEP 4: Learning parameters
%

printf('Learning parameters using training data...\n');
[softmaxModel] = softmaxTrain(theta,inputSize, numClasses, lambda, ...
                            train_data, train_labels, func,numiter); 

%%======================================================================
%% STEP 5: Testing
%

printf('Testing softmax model...\n');
[pred result] = softmaxPredict(softmaxModel, test_data);

acc = mean(test_labels(:) == pred(:));
printf('Accuracy: %0.3f%%\n', acc * 100);
end
