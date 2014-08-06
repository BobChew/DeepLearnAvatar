function [dW1 theta] = UFLDL_SparseAE(data,func,numiter,theta,hiddenSize,sparsityParam,beta,lambda)
%%======================================================================
%% STEP 0: Set parameters

printf('Running Sparse AutoEncoder using UFLDL...\n');
visibleSize = size(data,1);   % number of input units 

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
printf('Setup AutoEncoder...\n');

[cost, grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, data);
dW1 = reshape(grad(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%csvwrite('SAE_gradient_UF_r1.csv', dW1);

%%======================================================================
%% STEP 4: training sparse autoencoder

%%%%%options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
%%%%%options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
%%%%%options.display = 'on';
printf('Training...\n');

[opttheta, cost] = optim(@(p) sparseAutoencoderLinearCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, data), ...
			   theta, func, numiter);

%%%%%[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
%%%%%                                   visibleSize, hiddenSize, ...
%%%%%                                   lambda, sparsityParam, ...
%%%%%                                   beta, patches), ...
%%%%%                              theta, options);
%%%%%options = optimset('GradObj','on','MaxIter',600);
%%%%%[opttheta, cost] = ...
%%%%%	   fmincg(@(p)(sparseAutoencoderCost(p, visibleSize, hiddenSize, ...
%%%%%	   lambda, sparsityParam, beta, patches)), theta, options);

%%======================================================================
%% STEP 5: Visualization 
%printf('Visualization...\n');

%W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%theta = opttheta;
%display(W1);
%figure;
%display_network(W1', 12); 

%printf('end');
%print -djpeg weights.jpg   % save the visualization to a file 
end

