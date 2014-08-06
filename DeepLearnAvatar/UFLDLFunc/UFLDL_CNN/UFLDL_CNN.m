function [trainFeatures testFeatures result] = UFLDL_CNN(train_data,train_labels,test_data,test_labels,num_labels,func,initheta,numiter,hiddenSize,patchDim,poolDim,lambda,stepSize,W,b,ZCAWhite,meanPatch)

%%======================================================================
%% STEP 0: Initialization

imageDim = size(train_data, 1);         % image dimension
imageChannels = size(train_data, 3);     % number of channels (rgb, so 3)

numTrainImages = size(train_data,4);    % number of patches
numTestImages = size(test_data,4);

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units

%epsilon = 0.1;	       % epsilon for ZCA whitening

%%======================================================================
%% STEP 3: Convolve and pool with the dataset

assert(mod(hiddenSize, stepSize) == 0, 'stepSize should divide hiddenSize');

pooledFeaturesTrain = zeros(hiddenSize, numTrainImages, ...
    floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim) );
pooledFeaturesTest = zeros(hiddenSize, numTestImages, ...
    floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim) );

for convPart = 1:(hiddenSize / stepSize)
    
    featureStart = (convPart - 1) * stepSize + 1;
    featureEnd = convPart * stepSize;
    
    printf('Step %d: features %d to %d\n', convPart, featureStart, featureEnd);  
    Wt = W(featureStart:featureEnd, :);
    bt = b(featureStart:featureEnd);    
    
    printf('Convolving and pooling train images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
        train_data, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTrain(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
%    if (convPart == 1)
%	save('UF_c1.mat','convolvedFeaturesThis')
%	save('UF_s1.mat','pooledFeaturesThis')
%    end

    clear convolvedFeaturesThis pooledFeaturesThis;
    
    printf('Convolving and pooling test images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
        test_data, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTest(featureStart:featureEnd, :, :, :) = pooledFeaturesThis; 

    clear convolvedFeaturesThis pooledFeaturesThis;

end

%save('UFFeatures.mat', 'pooledFeaturesTrain', 'pooledFeaturesTest');

%%======================================================================
%% STEP 4: Use pooled features for classification

% Setup parameters for softmax
softmaxLambda = lambda;
numClasses = num_labels;
% Reshape the pooledFeatures to form an input vector for softmax
trainFeatures = permute(pooledFeaturesTrain, [1 3 4 2]);
%display(size(trainFeatures));
trainFeatures = reshape(trainFeatures, numel(pooledFeaturesTrain) / numTrainImages, numTrainImages);
initheta = initheta(1:size(trainFeatures, 1)*numClasses);
%softmaxY = train_labels;

%options = struct;
%options.maxIter = 200;
%softmaxModel = softmaxTrain(numel(pooledFeaturesTrain) / numTrainImages,...
%    numClasses, softmaxLambda, softmaxX, softmaxY, options);
softmaxModel = softmaxTrain(initheta, numel(pooledFeaturesTrain) / numTrainImages,...
    numClasses, softmaxLambda, trainFeatures, train_labels, func, numiter);

%%======================================================================
%% STEP 5: Test classifer
%  Now you will test your trained classifer against the test images

testFeatures = permute(pooledFeaturesTest, [1 3 4 2]);
testFeatures = reshape(testFeatures, numel(pooledFeaturesTest) / numTestImages, numTestImages);
%softmaxY = test_labels;

[pred result] = softmaxPredict(softmaxModel, testFeatures);
acc = (pred(:) == test_labels(:));
acc = sum(acc) / size(acc, 1);
printf('Accuracy: %2.3f%%\n', acc * 100);

% You should expect to get an accuracy of around 80% on the test images.
end
