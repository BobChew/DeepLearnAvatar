function [sae1 sae2 smnn finalnn result result_tune] = Toolbox_StackedAE(train_data,train_labels,test_data,test_labels,num_labels,func_SAE,func_SM,learningrate,numepochs_SAE,numepochs_SM,hiddenSizeL1,hiddenSizeL2,sparsity,beta,lambda_SAE,lambda_SM)

printf('Running Stacked AutoEncoder using DeepLearnToolbox...\n');

inputSize = size(train_data,2)

printf('Training the first layer Sparse Autoencoder...\n');

sae1 = Toolbox_SparseAE(train_data,func_SAE,learningrate,numepochs_SAE,hiddenSizeL1,sparsity,beta,lambda_SAE);

%sae1_theta = [sae1.ae{1}.W{1}(:);sae1.ae{1}.W{2}(:)];
%inithetaL1 = sae1.initial_theta;

printf('Training the second layer Sparse Autoencoder...\n');

ff1 = nnsetup([inputSize hiddenSizeL1]);
ff1.activation_funciton = 'sigm';
ff1.W{1} = sae1.ae{1}.W{1};
fake_output = zeros(size(train_data,1), hiddenSizeL1);
ff1 = nnff(ff1,train_data,fake_output);
sae1Features = ff1.a{2};

sae2 = Toolbox_SparseAE(sae1Features,func_SAE,learningrate,numepochs_SAE,hiddenSizeL2,sparsity,beta,lambda_SAE);

%sae2_theta = [sae2.ae{1}.W{1}(:);sae2.ae{1}.W{2}(:)];
%inithetaL2 = sae2.initial_theta;

printf('Training the Softmax classifier...\n');

ff2 = nnsetup([hiddenSizeL1 hiddenSizeL2]);
ff2.activation_funciton = 'sigm';
ff2.W{1} = sae2.ae{1}.W{1};
fake_output = zeros(size(sae1Features,1), hiddenSizeL2);
ff2 = nnff(ff2,sae1Features,fake_output);
sae2Features = ff2.a{2};

[smnn acc result] = Toolbox_Softmax(sae2Features,train_labels,sae2Features,train_labels,func_SM,learningrate,numepochs_SM,num_labels,lambda_SM,0,magic(3));

%sm_theta = smnn.W{1}(:);
%initheta_SM = smnn.initial_theta;

printf('Finetune softmax model and Testing...\n');

finalnn = nnsetup([inputSize, hiddenSizeL1, hiddenSizeL2, num_labels]);
finalnn.W{1} = sae1.ae{1}.W{1};
finalnn.W{2} = sae2.ae{1}.W{1};
finalnn.W{3} = smnn.W{1};
finalnn.activation_function = 'sigm';
finalnn.learningRate = learningrate;
finalnn.weightPenaltyL2 = lambda_SAE;
finalnn.output = 'softmax';
opts.numepochs = numepochs_SAE;
opts.batchsize = size(train_data, 1);
groundTruth = full(sparse(1:size(train_labels, 1), train_labels, 1));
testTruth = full(sparse(1:size(test_labels, 1), test_labels, 1));

[er, bad] = nntest(finalnn, test_data, testTruth);
[test_l,result] = nnpredict(finalnn, test_data);
acc = 1-er;
printf('Test Accuracy before finetuning is %0.3f%%\n', acc*100);

finalnn = nntrain(finalnn, train_data, groundTruth, opts);

[er_tune, bad_tune] = nntest(finalnn, test_data, testTruth);
[test_l_tune,result_tune] = nnpredict(finalnn, test_data);
acc_tune = 1-er_tune;
printf('Test Accuracy after finetuning is %0.3f%%\n', acc_tune*100);

end
