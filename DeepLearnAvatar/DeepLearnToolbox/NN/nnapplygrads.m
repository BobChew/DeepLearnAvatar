function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0)
%	    display(i);
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
%	    dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(1, size(nn.W{i}, 2)); nn.W{i}];
        else
            dW = nn.dW{i};
        end
       
%	display(nn.dW{2});
%	display((nn.dW{1} + nn.weightPenaltyL2 * nn.W{1})(1:5,1:5));
%	if(i==2) display(dW); end; 
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end
end
