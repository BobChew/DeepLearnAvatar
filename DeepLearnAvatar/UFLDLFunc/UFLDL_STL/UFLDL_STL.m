function [dW1 result] = UFLDL_STL(train_data,train_labels,test_data,test_labels,unlabeled_data,num_labels,func_SAE,func_SM,theta_SAE,theta_SM,numiter_SAE,numiter_SM,hiddenSize,sparsityParam,beta,lambda)

%% ======================================================================
%  STEP 2: Train the sparse autoencoder

printf('Train the sparse autoencoder using UFLDL...\n');

opttheta = theta_SAE; 
visibleSize = size(unlabeled_data,1);

[cost, grad] = sparseAutoencoderCost(theta_SAE, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, unlabeled_data);
dW1 = reshape(grad(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%csvwrite('SAE_gradient_UF_r1.csv', dW1);

%options.Method = 'lbfgs';   
%options.maxIter = 400;    % Maximum number of iterations of L-BFGS to run   
%options.display = 'on';  
%[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...  
%                                   visibleSize, hiddenSize, ...  
%                                   lambda, sparsityParam, ...  
%                                   beta, unlabeledData), ...  
%                              theta, options);

[opttheta, cost] = optim(@(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeled_data), ...
                           theta_SAE, func_SAE, numiter_SAE);


printf('Finishing training the sparse autoencoder.\n');

%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * visibleSize), hiddenSize, visibleSize);
figure;
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  

printf('Extract features from the supervised dataset...\n');

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, visibleSize, ...
                                       train_data);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, visibleSize, ...
                                       test_data);

%%======================================================================
%% STEP 4: Train the softmax classifier

printf('Train the sofmax classifier using UFLDL...\n');

%softmaxModel = struct;  

[inputSize,sampleN] = size(trainFeatures);
theta_SM  = theta_SM(1:hiddenSize*num_labels);

[cost, grad] = softmaxCost(theta_SM, num_labels, hiddenSize, lambda, trainFeatures, train_labels);
dW1 = reshape(grad(1:num_labels*hiddenSize), num_labels, hiddenSize);
%csvwrite('Softmax_gradient_UF_r1.csv', dW1);                                    

[softmaxModel] = softmaxTrain(theta_SM,hiddenSize, num_labels, lambda, trainFeatures, train_labels, func_SM,numiter_SM);

%%======================================================================
%% STEP 5: Testing 

printf('Testing...\n');

[pred result] = softmaxPredict(softmaxModel, testFeatures);


%% -----------------------------------------------------

% Classification Score
printf('Test Accuracy: %f%%\n', 100*mean(pred(:) == test_labels(:)));
%
end 
