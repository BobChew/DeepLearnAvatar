function [trainFeatures testFeatures nn result] = Toolbox_CNN(train_data,train_labels,test_data,test_labels,num_labels,func,learningrate,numepochs,hiddenSize,patchDim,poolDim,lambda,stepSize,W,b)

printf('Train Convolutional Neural Network using DeepLearnToolbox...\n');
imageDim = size(train_data, 1);
imageChannels = size(train_data, 3);
numTrain = size(train_data, 4);
numTest = size(test_data, 4);
patchSize = patchDim*patchDim;

pooledFeaturesTrain = zeros(hiddenSize, floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim), numTrain);
pooledFeaturesTest = zeros(hiddenSize, floor((imageDim - patchDim + 1) / poolDim), ...
    floor((imageDim - patchDim + 1) / poolDim), numTest);

printf('Setting up CNN...\n');

cnn.layers = {
	struct('type', 'i')
	struct('type', 'c', 'outputmaps', 1, 'kernelsize', patchDim)
	struct('type', 's', 'scale', poolDim)
}
cnn.inputmaps = imageChannels;
opts.alpha = 1;
opts.numepochs = 1;
cnn_train = cnnsetup(cnn, squeeze(train_data(:,:,1,:)), train_labels);
cnn_test = cnnsetup(cnn, squeeze(test_data(:,:,1,:)), test_labels);

for convPart = 1:(hiddenSize / stepSize)

	featureStart = (convPart-1)*stepSize+1;
	featureEnd = convPart*stepSize;
	Wt = W(featureStart:featureEnd,:);
	bt = b(featureStart:featureEnd);

	for featureNum = 1:stepSize

		for channel = 1:imageChannels
			offset = (channel-1)*patchSize;
			feature = reshape(Wt(featureNum,offset+1:offset+patchSize), patchDim, patchDim);
			feature = flipud(fliplr(squeeze(feature)));
%			train_rgb = squeeze(train_data(:,:,channel,:));
%			test_rgb = squeeze(test_data(:,:,channel,:));

			cnn_train.layers{2}.k{channel}{1} = feature;
			cnn_test.layers{2}.k{channel}{1} = feature;
		end

		cnn_train.layers{2}.b{1} = bt(featureNum);
%		cnn_train.layers{3}.b{1} = bt(featureNum);
		cnn_train = cnnff(cnn_train,train_data);
		pooledFeaturesTrain(featureStart+featureNum-1, :, :, :) = cnn_train.layers{3}.a{1};

%		if (convPart == 1 && featureNum == 1)
%			TB_c1 = cnn_train.layers{2}.a{1};
%			TB_s1 = cnn_train.layers{3}.a{1};
%			save('TB_c1.mat','TB_c1')
%			save('TB_s1.mat','TB_s1')
%		end

                cnn_test.layers{2}.b{1} = bt(featureNum);
%                cnn_test.layers{3}.b{1} = bt(featureNum);
                cnn_test = cnnff(cnn_test,test_data);
		pooledFeaturesTest(featureStart+featureNum-1, :, :, :) = cnn_test.layers{3}.a{1};

	end
end

%save('TBFeatures.mat', 'pooledFeaturesTrain', 'pooledFeaturesTest');

printf('Train the sofmax classifier...\n');

%trainFeatures = permute(pooledFeaturesTrain, [1 3 4 2]);
%display(size(pooledFeaturesTrain));
trainFeatures = reshape(pooledFeaturesTrain, numel(pooledFeaturesTrain)/numTrain,numTrain);
%testFeatures = permute(pooledFeaturesTest, [1 3 4 2]);
testFeatures = reshape(pooledFeaturesTest, numel(pooledFeaturesTest)/numTest,numTest);
[nn acc result] = Toolbox_Softmax(trainFeatures',train_labels,testFeatures',test_labels,func,learningrate,numepochs,num_labels,lambda,0,magic(3));

printf('Test Accuracy: %f%%\n', 100*acc);

end
