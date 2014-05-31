%function [theta_TB, theta_UF] = train_SAE(filename)
function train_Softmax(filename)

func = 'sigm';			%activation funcion
learningrate = 15;
train_percentage = 0.7;		%percentage of data used for training, the rest of data is used for testing
numClasses = 5;		%number of clusters
numepochs = 1;			%number of iterations using Toolbox
lambda = 0.0001;
func_UF = 'fmincg';		%training function using UFLDL
numiter = 0;			%number of iterations using UFLDL

data = csvread(filename);

num_train = floor(size(data,1)*train_percentage);
train_data = data(1:num_train,1:size(data,2)-1);
train_labels = data(1:num_train,end);
test_data = data(num_train+1:end,1:size(data,2)-1);
test_labels = data(num_train+1:end,end);

rand('seed',0);
[acc_TB result_TB] = Toolbox_Softmax(train_data,train_labels,test_data,test_labels,func,learningrate,numepochs,numClasses,lambda);
%disp(size(result_TB));
csvwrite('Softmax_result_TB.csv',result_TB);

initheta = csvread('initialtheta.csv');
%initheta = initheta(1:size(train_data,2)*numClasses);


rand('seed',0);
[acc_UF result_UF] = UFLDL_Softmax(train_data,train_labels,test_data,test_labels,func_UF,numiter,numClasses,initheta,lambda);
%disp(size(theta_UF));
csvwrite('Softmax_result_UF.csv',result_UF);
end
