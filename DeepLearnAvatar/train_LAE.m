%function [theta_TB, theta_UF] = train_SAE(filename)
function train_LAE(filename)

func = 'sigm';			%activation funcion
learningrate = 15;
numepochs = 1;		%number of iterations using Toolbox
hiddenSize = 25;
sparsity = 0.01;
beta = 3;
lambda = 0.0001;
func_UF = 'fmincg';		%training function using UFLDL
numiter = 1;			%number of iterations using UFLDL

data = csvread(filename);

rand('seed',0);
sae_TB = Toolbox_LAE(data',func,learningrate,numepochs,hiddenSize,sparsity,beta,lambda);
%disp(size(theta_TB));
theta_TB = [sae_TB.ae{1}.W{1}(:);sae_TB.ae{1}.W{2}(:)];\
gradient_TB = sae_TB.ae{1}.dW_r1;
csvwrite('output/LAE_Theta_TB.csv',theta_TB);
csvwrite('output/LAE_GradientR1_TB.csv',gradient_TB);

initheta = sae_TB.ae{1}.initial_theta;

rand('seed',0);
[dW1 vtheta_UF] = UFLDL_LAE(data,func_UF,numiter,initheta,hiddenSize,sparsity,beta,lambda);
%disp(size(theta_UF));
csvwrite('output/LAE_theta_UF.csv',theta_UF);
csvwrite('output/LAE_GradientR1_UF.csv',dW1);
end
