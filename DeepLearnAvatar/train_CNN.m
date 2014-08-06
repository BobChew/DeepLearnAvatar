%function train_CNN(filename_train,filename_test,filename_features)
function train_CNN

num_labels = 4;
func_TB = 'sigm';			%activation funcion
learningrate = 5;
numepochs = 1;		%number of iterations using Toolbox
hiddenSize = 400;
patchDim = 8;
%numPatches = 50000;
poolDim = 19;
lambda = 0.0001;
stepSize = 50;			% number of features to be convolved at a time

func_UF = 'lbfgs';		%training function using UFLDL
numiter = 1;

%load filename_train;
%load filename_test;
%load filename_features;
load data/stlTrainSubset.mat
load data/stlTestSubset.mat
load data/STL10Features.mat

imageDim = size(trainImages, 1);
imageChannels = size(trainImages, 3);
visibleSize = patchDim * patchDim * imageChannels;
outputSize = visibleSize;

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
WT = W*ZCAWhite;
b_mean = b - WT*meanPatch;


rand('seed',0);
[trainFeatures_TB testFeatures_TB nn_TB result_TB] = Toolbox_CNN(trainImages,trainLabels,testImages,testLabels,num_labels,func_TB,learningrate,numepochs,hiddenSize,patchDim,poolDim,lambda,stepSize,WT,b_mean);
%disp(size(theta_TB));T
csvwrite('output/CNN_TrainFeatures_TB.csv',trainFeatures_TB);
csvwrite('output/CNN_TestFeatures_TB.csv',testFeatures_TB);
csvwrite('output/CNN_Result_TB.csv',result_TB);

initheta = nn_TB.initial_theta;

rand('seed',0);
[trainFeatures_UF testFeatures_UF result_UF] = UFLDL_CNN(trainImages,trainLabels,testImages,testLabels,num_labels,func_UF,initheta,numiter,hiddenSize,patchDim,poolDim,lambda,stepSize,W,b,ZCAWhite,meanPatch);
%disp(size(theta_UF));
csvwrite('output/CNN_TrainFeatures_UF.csv',trainFeatures_UF);
csvwrite('output/CNN_TestFeatures_UF.csv',testFeatures_UF);
csvwrite('output/CNN_Result_UF.csv',result_UF);
end
