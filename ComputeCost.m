function [ cost ] = ComputeCost(output_error, weights_array, ...
                                num_data_samples, num_layers, lambda)
%COMPUTECOST calculates cost of hypothesis using an l2-norm with 
% regularization. output_error is the difference between our reference data 
% and our hypothesis.

cost = 1/(2*num_data_samples)*output_error*output_error';

% Adding regularization
for layer = 1:num_layers-1
    cost = cost + (weights_array{layer}(:)'*weights_array{layer}(:))...
                   *lambda/(2*num_data_samples);
end

end

