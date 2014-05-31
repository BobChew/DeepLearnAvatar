function theta = UFLDL_SAE(data,func,numiter,theta,hiddenSize,sparsityParam,beta,lambda)
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

printf('Running Sparse AutoEncoder using UFLDL...\n');
visibleSize = size(data,2);   % number of input units 
%hiddenSize = 25;     % number of hidden units 
%sparsityParam = 0.01;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
%lambda = 0.0001;     % weight decay parameter       
%beta = 3;            % weight of sparsity penalty term

%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
printf('Setup AutoEncoder...\n');
%patches = csvread('patches.csv');
data = data';
 
% We are using display_network from the autoencoder code
%%%%%display_network(patches(:,1:100)); % Show the first 100 images
%disp(labels(1:10));

%  Obtain random parameters theta
%theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost
%

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta, data);
dW1 = reshape(grad(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
csvwrite('gradient_UF_r1.csv', dW1);

%%======================================================================
%% STEP 3: Gradient Checking
%

%%%%%checkNumericalGradient();


% for the sparse autoencoder.  
%%%%%numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
%%%%%                                                  hiddenSize, lambda, ...
%%%%%                                                  sparsityParam, beta, ...
%%%%%                                                  patches), theta);

% Use this to visually compare the gradients side by side
%%%%%disp([numgrad grad]);
%%%%%bobgrad = [numgrad grad];
%%%%%save bobgradcheck.mat bobgrad;

% Compare numerically computed gradients with the ones obtained from backpropagation
%%%%%diff = norm(numgrad-grad)/norm(numgrad+grad);
%%%%%disp(diff); % Should be small. In our implementation, these values are
            % usually less than 1e-9.

            % When you got this working, Congratulations!!! 

%%======================================================================
%% STEP 4: After verifying that your implementation of
%  sparseAutoencoderCost is correct, You can start training your sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
%%%%%theta = initializeParameters(hiddenSize, visibleSize);

%%%%%options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
%%%%%options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
%%%%%options.display = 'on';
printf('Training...\n');

[opttheta, cost] = optim(@(p) sparseAutoencoderCost(p, ...
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
printf('Visualization...\n');

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
theta = W1;
%display(W1);
figure;
display_network(W1', 12); 

%printf('end');
%print -djpeg weights.jpg   % save the visualization to a file 
end

