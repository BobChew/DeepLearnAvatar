function [dW1 result result_tune] = UFLDL_StackedAE(trainData,trainLabels,testData,testLabels,numClasses,func_SAE,func_SM,initheta_SAE1,initheta_SAE2,initheta_SM,numiter_SAE,numiter_SM,hiddenSizeL1,hiddenSizeL2,sparsityParam,beta,lambda,softmaxLambda)
%%======================================================================
%% STEP 0: Set parameters

inputSize = size(trainData,1);

%%======================================================================
%% STEP 2: Train the first sparse autoencoder

[cost, grad] = sparseAutoencoderCost(initheta_SAE1, inputSize, hiddenSizeL1, lambda, ...
                                     sparsityParam, beta, trainData);
dW1 = reshape(grad(1:hiddenSize*inputSize), hiddenSize, inputSize);
%csvwrite('SAE_gradient_UF_r1.csv', dW1);

%options = struct;
%options.Method = 'lbfgs';
%options.maxIter = 400;
%options.display = 'on';
%[sae1OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
%    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),...
%    sae1Theta,options);
[sae1OptTheta, cost] = optim(@(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                           initheta_SAE1, func_SAE, numiter_SAE);

W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
display_network(W1');

%%======================================================================
%% STEP 2: Train the second sparse autoencoder

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%[sae2OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
%    hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2Theta,options);
[sae2OptTheta, cost] = optim(@(p) sparseAutoencoderCost(p, ...
                                    hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                           initheta_SAE2, func_SAE, numiter_SAE);
%figure;
%W11 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
%W12 = reshape(sae2OptTheta(1:hiddenSizeL2 * hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
%  display_network(log(W11' ./ (1-W11')) * W12');
%   W12_temp = W12(1:196,1:196);
%   display_network(W12_temp');
%   figure;
%   display_network(W12_temp');

%%======================================================================
%% STEP 3: Train the softmax classifier

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

initheta_SM  = initheta_SM(1:hiddenSizeL2*numClasses);
softmaxModel = softmaxTrain(initheta_SM,hiddenSizeL2,numClasses,softmaxLambda,...
                            sae2Features,trainLabels,func_SM,numiter_SM);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);

%%======================================================================
%% STEP 5: Finetune softmax model

stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
%                         numClasses, netconfig,lambda, trainData, trainLabels),...
%                        stackedAETheta,options);
[stackedAEOptTheta, cost] = optim(@(p)stackedAECost(p, inputSize, hiddenSizeL2,...
                                   numClasses, netconfig, lambda, ...
                                   trainData, trainLabels), ...
                           stackedAETheta, func_SAE, numiter_SAE);
%figure;
%  optStack = params2stack(stackedAEOptTheta(hiddenSizeL2*numClasses+1:end), netconfig);
%  W11 = optStack{1}.w;
%  W12 = optStack{2}.w;
  % TODO(zellyn): figure out how to display a 2-level network
  % display_network(log(1 ./ (1-W11')) * W12');

%%======================================================================
%% STEP 6: Test 

[pred result] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
printf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred_tune result_tune] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc_tune = mean(testLabels(:) == pred_tune(:));
printf('After Finetuning Test Accuracy: %0.3f%%\n', acc_tune * 100);

end
