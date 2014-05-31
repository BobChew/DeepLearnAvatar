function [acc result] = UFLDL_Softmax(train_data,train_labels,test_data,test_labels,func,numiter,numClasses,theta,lambda)

printf('Running Softmax using UFLDL...\n');
%%======================================================================
%% STEP 0: Initialise constants and parameters
%
%  Here we define and initialise some constants which allow your code
%  to be used more generally on any arbitrary input. 
%  We also initialise some parameters used for tuning the model.

%inputSize = 28 * 28; % Size of input vector (MNIST images are 28x28)
inputSize = size(train_data,2);

%%======================================================================
%% STEP 1: Load data
%
printf('Normalizing trainging data...\n');
%%images = loadMNISTImages('MNIST/train-images-idx3-ubyte');
%%labels = loadMNISTLabels('MNIST/train-labels-idx1-ubyte');
[train_data, mu, sigma] = zscore(train_data);   %  mean & max
test_data = normalize(test_data, mu, sigma);
%train_labels(train_labels==0) = 10; % Remap 0 to 10
%test_labels(test_labels==0) = 10;
train_data = train_data';
test_data = test_data';

% For debugging purposes, you may wish to reduce the size of the input data
% in order to speed up gradient checking. 
% Here, we create synthetic dataset using random data for testing

% Randomly initialise theta
%%theta = 0.005 * randn(numClasses * inputSize, 1);

%%======================================================================
%% STEP 2: Implement softmaxCost
%
%  Implement softmaxCost in softmaxCost.m. 

[cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, train_data, train_labels);
dW1 = reshape(grad(1:numClasses*inputSize), numClasses, inputSize);
csvwrite('gradient_UF_r1.csv', dW1);                                     

%%======================================================================
%% STEP 4: Learning parameters
%

printf('Learning parameters using training data...\n');
softmaxModel = softmaxTrain(theta,inputSize, numClasses, lambda, ...
                            train_data, train_labels, func,numiter); 

%%======================================================================
%% STEP 5: Testing
%

%%images = loadMNISTImages('MNIST/t10k-images-idx3-ubyte');
%%labels = loadMNISTLabels('MNIST/t10k-labels-idx1-ubyte');

% You will have to implement softmaxPredict in softmaxPredict.m

printf('Testing softmax model...\n');
[pred result] = softmaxPredict(softmaxModel, test_data);

acc = mean(test_labels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
