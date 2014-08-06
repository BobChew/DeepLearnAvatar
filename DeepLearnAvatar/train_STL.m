%function [theta_TB, theta_UF] = train_SAE(filename)
function train_STL(filename_data,filename_labels)

num_labels = 5;
func_SAE_TB = 'sigm';			%activation funcion
func_SM_TB = 'sigm';
learningrate = 5;
numepochs_SAE = 1;		%number of iterations using Toolbox
numepochs_SM = 1;
hiddenSize = 200;
sparsity = 0.01;
beta = 3;
lambda = 0.003;

func_SAE_UF = 'lbfgs';		%training function using UFLDL
func_SM_UF = 'lbfgs';
numiter_SAE = 1;			%number of iterations using UFLDL
numiter_SM = 1;

mnistData   = loadMNISTImages(filename_data);
mnistLabels = loadMNISTLabels(filename_labels);

labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
unlabeledSet = find(mnistLabels >= 5);

numTrain = round(numel(labeledSet)/2);
trainSet = labeledSet(1:numTrain);
testSet  = labeledSet(numTrain+1:end);

unlabeledData = mnistData(:, unlabeledSet);

trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

%disp(size(trainData));
%disp(size(trainLabels));

rand('seed',0);
[sae_TB nn_TB result_TB] = Toolbox_STL(trainData',trainLabels',testData',testLabels',unlabeledData',num_labels,func_SAE_TB,func_SM_TB,learningrate,numepochs_SAE,numepochs_SM,hiddenSize,sparsity,beta,lambda);
%disp(size(theta_TB));
gradient_TB = sae_TB.ae{1}.dW_r1;
csvwrite('output/STL_Result_TB.csv',result_TB);
csvwrite('output/STL_GradientR1_TB.csv',gradient_TB);

%initheta = csvread('initialtheta.csv');
%display(sae_TB.ae{1}.initial_theta);
initheta_SAE = sae_TB.ae{1}.initial_theta;
initheta_SM = nn_TB.initial_theta;
%disp(initheta_SAE(1:10));

%disp('flag');
rand('seed',0);
[dW1 result_UF] = UFLDL_STL(trainData,trainLabels,testData,testLabels,unlabeledData,num_labels,func_SAE_UF,func_SM_UF,initheta_SAE,initheta_SM,numiter_SAE,numiter_SM,hiddenSize,sparsity,beta,lambda);
%disp(size(theta_UF));
csvwrite('output/STL_Result_UF.csv',result_UF);
csvwrite('output/STL_GradientR1_UF.csv',dW1);
end
