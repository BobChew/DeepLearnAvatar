function test_example_SAE
load mnist_uint8;

train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
printf('Step 1\n');
rand('state',0)
sae = saesetup([784 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
printf('Flag 1\n');
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
printf('Step 2\n');
nn = nnsetup([784 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
printf('Step 3\n');
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
printf('Flag 2\n');
[er, bad] = nntest(nn, test_x, test_y);
printf('Flag 3\n');
assert(er < 0.16, 'Too big error');
printf('Flag 4\n');
