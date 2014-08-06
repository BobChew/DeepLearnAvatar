function train_Softmax(trainname, trainlabel, testname, testlabel)

func = 'none';			%activation funcion
learningrate = 1;
train_percentage = 1;		%percentage of data used for training, the rest of data is used for testing
numClasses = 10;		%number of clusters
numepochs = 1;			%number of iterations using Toolbox
lambda = 0.0001;
func_UF = 'fmincg';		%training function using UFLDL
numiter = 1;			%number of iterations using UFLDL

%data = csvread(filename);

%num_train = floor(size(data,1)*train_percentage);
%train_data = data(1:num_train,1:size(data,2)-1);
%train_labels = data(1:num_train,end);
%test_data = data(num_train+1:end,1:size(data,2)-1);
%test_labels = data(num_train+1:end,end);
train_data = loadMNISTImages(trainname);
train_labels = loadMNISTLabels(trainlabel);
test_data = loadMNISTImages(testname);
test_labels = loadMNISTLabels(testlabel);
train_labels(train_labels==0) = 10; % Remap 0 to 10
test_labels(test_labels==0) = 10; % Remap 0 to 10


rand('seed',0);
[nn_TB acc_TB result_TB] = Toolbox_Softmax(train_data',train_labels,test_data',test_labels,func,learningrate,numepochs,numClasses,lambda,0,magic(3));
%disp(size(result_TB));
gradient_TB = nn_TB.dW_r1;
csvwrite('output/Softmax_Result_TB.csv',result_TB);
csvwrite('output/Softmax_GradientR1_TB.csv',gradient_TB);

initheta = nn_TB.initial_theta;
%initheta = initheta(1:size(train_data,2)*numClasses);

[dW1 acc_UF result_UF] = UFLDL_Softmax(train_data',train_labels,test_data',test_labels,func_UF,numiter,numClasses,initheta,lambda);
%disp(size(result_UF));
csvwrite('output/Softmax_Result_UF.csv',result_UF);
csvwrite('output/Softmax_GradientR1_UF.csv',dW1);
end
