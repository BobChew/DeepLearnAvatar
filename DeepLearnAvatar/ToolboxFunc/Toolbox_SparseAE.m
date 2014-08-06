function sae = Toolbox_SparseAE(data,func,learningrate,numepochs,hiddenSize,sparsity,beta,lambda)

printf('Running Sparse AutoEncoder using DeepLearnToolbox...\n');
%patches = loadMNISTImages('train-images.idx3-ubyte');i
%patches = csvread('patches.csv');	%numofrecords*numoffeatures
%patches = patches';

printf('Setup AutoEncoder...\n');
sae = saesetup([size(data,2) hiddenSize]);	% [visibleSize, hiddenSize]
sae.ae{1}.activation_function = func;		% sigmoid function
sae.ae{1}.learningRate = learningrate;
sae.ae{1}.scaling_learningRate = 1;
%sae.ae{1}.momentum = 0.5;
sae.ae{1}.sparsityTarget = sparsity;
%%sae.ae{1}.inputZeroMaskedFraction = 0;
sae.ae{1}.nonSparsityPenalty = beta;
sae.ae{1}.weightPenaltyL2 = lambda;		%
opts.numepochs = numepochs;
opts.batchsize = size(data,1);

printf('Training...\n');
sae = saetrain(sae, data, opts);

%theta = sae.ae{1}.W{1}(:,2:end);
%theta = [sae.ae{1}.W{1}(:);sae.ae{1}.W{2}(:)];

printf('Visualization...\n');
%display(size(sae.ae{1}.W{1}));
%visualize(sae.ae{1}.W{1}(:,2:end)');
figure;
display_network(sae.ae{1}.W{1}(:,2:end)', 12);
end
