function [theta, cost] = optim(f, X, method, iter)

%genpath(".")
%addpath(genpath("."))

if strcmp(method,'lbfgs')
	options.Method = 'lbfgs';
	options.maxIter = iter;
	options.display = 'on';

	[theta cost] = minFunc(f, X, options);
elseif strcmp(method,'fmincg')
	options = optimset('GradObj','on','MaxIter',iter);
	[theta cost] = fmincg(f,X,options);
else;
end
				     
