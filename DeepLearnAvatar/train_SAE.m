%function [theta_TB, theta_UF] = train_SAE(filename)
function train_SAE(filename)

func = 'sigm';			%activation funcion
learningrate = 15;
numepochs = 1;		%number of iterations using Toolbox
hiddensize = 25;
sparsity = 0.01;
beta = 3;
lambda = 0.0001;
func_UF = 'fmincg';		%training function using UFLDL
numiter = 0;			%number of iterations using UFLDL

data = csvread(filename);

rand('seed',0);
theta_TB = Toolbox_SAE(data,func,learningrate,numepochs,hiddensize,sparsity,beta,lambda);
%disp(size(theta_TB));
csvwrite('SAE_theta_TB.csv',theta_TB);

initheta = csvread('initialtheta.csv');

rand('seed',0);
theta_UF = UFLDL_SAE(data,func_UF,numiter,initheta,hiddensize,sparsity,beta,lambda);
%disp(size(theta_UF));
csvwrite('SAE_theta_UF.csv',theta_UF);
end
