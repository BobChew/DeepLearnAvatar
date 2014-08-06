function [sae nn result] = Toolbox_STL(train_data,train_labels,test_data,test_labels,unlabeled_data,num_labels,func_SAE,func_SM,learningrate,numepochs_SAE,numepochs_SM,hiddenSize,sparsity,beta,lambda)

printf('Train Sparse AutoEncoder using DeepLearnToolbox...\n');

num_features = size(train_data,2);
sae = Toolbox_SparseAE(unlabeled_data,func_SAE,learningrate,numepochs_SAE,hiddenSize,sparsity,beta,lambda);

%train_theta = [sae_TB.ae{1}.W{1}(:);sae_TB.ae{1}.W{2}(:)];
%initheta = sae_TB.ae{1}.initial_theta;

printf('Extract features from the supervised dataset...\n');

ff = nnsetup([size(unlabeled_data,2) hiddenSize]);
ff.activation_funciton = 'sigm';
ff.W{1} = sae.ae{1}.W{1};
fake_output = zeros(size(train_data,1), hiddenSize);
ff = nnff(ff,train_data,fake_output);
trainFeatures = ff.a{2};
fake_output = zeros(size(test_data,1), hiddenSize);
ff = nnff(ff,test_data,fake_output);
testFeatures = ff.a{2};

printf('Train the sofmax classifier...\n');

[nn acc result] = Toolbox_Softmax(trainFeatures,train_labels,testFeatures,test_labels,func_SM,learningrate,numepochs_SM,num_labels,lambda,0,magic(3));

printf('Test Accuracy: %f%%\n', 100*acc);

end
