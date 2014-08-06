%function [theta_TB, theta_UF] = train_SAE(filename)
function train_StackedAE(train_data,train_labels,test_data,test_labels)

num_labels = 10;
func_SAE_TB = 'sigm';			%activation funcion
func_SM_TB = 'sigm';
learningrate = 5;
numepochs_SAE = 1;		%number of iterations using Toolbox
numepochs_SM = 1;
hiddenSizeL1 = 200;
hiddenSizeL2 = 200;
sparsity = 0.1;
beta = 3;
lambda_SAE = 0.003;
lambda_SM = 0.0001;

func_SAE_UF = 'lbfgs';		%training function using UFLDL
func_SM_UF = 'lbfgs';
numiter_SAE = 1;			%number of iterations using UFLDL
numiter_SM = 1;

trainData   = loadMNISTImages(train_data);
trainLabels = loadMNISTLabels(train_labels);
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

testData   = loadMNISTImages(test_data);
testLabels = loadMNISTLabels(test_labels);
testLabels(testLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

%disp(size(trainData));
%disp(size(trainLabels));

rand('seed',0);
[sae1_TB sae2_TB smnn_TB finalnn_TB result_TB result_tune_TB] = Toolbox_StackedAE(trainData',trainLabels',testData',testLabels',num_labels,func_SAE_TB,func_SM_TB,learningrate,numepochs_SAE,numepochs_SM,hiddenSizeL1,hiddenSizeL2,sparsity,beta,lambda_SAE,lambda_SM);
%disp(size(theta_TB));
gradient_TB = sae1_TB.ae{1}.dW_r1;
csvwrite('output/StackedAE_Result_TB.csv',result_TB);
csvwrite('output/StackedAE_TunedResult_TB.csv',result_tune_TB);
csvwrite('output/StackedAE_GradientR1_TB.csv',gradient_TB);

%initheta = csvread('initialtheta.csv');
initheta_SAE1 = sae1_TB.ae{1}.initial_theta;
initheta_SAE2 = sae2_TB.ae{1}.initial_theta;
initheta_SM = smnn_TB.initial_theta;
%disp(initheta_SAE(1:10));

%disp('flag');
rand('seed',0);
[dW1 result_UF result_tune_UF] = UFLDL_StackedAE(trainData,trainLabels,testData,testLabels,num_labels,func_SAE_UF,func_SM_UF,initheta_SAE1,initheta_SAE2,initheta_SM,numiter_SAE,numiter_SM,hiddenSizeL1,hiddenSizeL2,sparsity,beta,lambda_SAE,lambda_SM);
%disp(size(theta_UF));
csvwrite('output/StackedAE_Result_UF.csv',result_UF);
csvwrite('output/StackedAE_TunedResult_UF.csv',result_tune_UF);
csvwrite('output/StackedAE_GradientR1_UF.csv',dW1);
end
