function [nn acc result] = Toolbox_Softmax(train_data,train_labels,test_data,test_labels,func,learning_rate,num_iter,num_labels,weight_penalty,initialize,initial_theta)

printf('Running Softmax using DeepLearnToolbox...\n');

% normalize
printf('Normalizing training data...\n');
[train_data, mu, sigma] = zscore(train_data); 	%  mean & max
test_data = normalize(test_data, mu, sigma);
%train_labels(train_labels==0) = 10; % Remap 0 to 10
%test_labels(test_labels==0) = 10;

printf('Setup Neural Network...\n');
[num_trains, visibleSize] = size(train_data);
nn = nnsetup([visibleSize, num_labels]);
if (initialize == 1)
	nn.W{1} = initial_theta;
end

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = learning_rate;
nn.weightPenaltyL2 = weight_penalty;
nn.output = 'softmax';
opts.numepochs =  num_iter;   %  Number of full sweeps through data
opts.batchsize = num_trains;  %  Take a mean gradient step over this many samples

printf('Learning parameters using training data...\n');
groundTruth = full(sparse(1:size(train_labels, 1), train_labels, 1));
if(size(groundTruth, 2) < num_labels)
	groundTruth = [groundTruth zeros(num_trains, 1)];
end
nn = nntrain(nn, train_data, groundTruth, opts);

printf('Testing softmax model...\n');
testTruth = full(sparse(1:size(test_labels, 1), test_labels, 1));
[er, bad] = nntest(nn, test_data, testTruth);
[test_l,result] = nnpredict(nn, test_data);

%assert(er < 0.08, 'Too big error');
acc = 1-er;
printf('Training Accuracy is %0.3f%%\n', acc*100);

%result = nn.a{end};

end
