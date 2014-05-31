Instructions using DeepLearnAvatar:

1. Open matlab/octave
2. Add path. Type:
	addpath(genpath('DeepLearnToolbox'));
	addpath(genpath('ToolboxFunc'));
	addpath(genpath('UFLDLFunc'));
3. Type:
	train_SAE(filename);
   or
	train_Softmax(filename);
   to run sparse auto-encoder or softmax classifier
4. Tune configuration parameters by modifying train_SAE.m and train_Softmax.m
5. Output for sparse auto-encoder:
	gradient_TB_r1.csv	gradient of the first iteration training SAE with Toolbox
	gradient_UF_r1.csv	gradient of the first iteration training SAE with UFLDL
				(the values in these two files should be close)
	SAE_theta_TB.csv	matrix of theta for the first lair of SAE with Toolbox
	SAE_theta_UF.csv	matrix of theta for the first lair of SAE with UFLDL
6. Output for Softmax classifier:
	gradient_TB_r1.csv	gradient of the first iteration training Softmax with Toolbox
	gradient_UF_r1.csv	gradient of the first iteration training Softmax with UFLDL
				(the values in these two files should be close)
	Softmax_result_TB.csv	result table of testset prediction with Toolbox
	Softmax_result_UF.csv	result table of testset prediction with UFLDL