% COMPUTECOST calculates cost of hypothesis using a cross-entropy cost 
% function with regularization. 
% - Parameters -
% h: model predictions in binary array format
% y: training/CV/test set labels in binary array format
% weights_array: cell array of neural network weights
% num_data_samples: number of training/CV/test set data points
% num_layers: total number of layers in neural network
% lambda: regularization parameter
% - Returns -
% cost: numerical value of cost

function [ cost ] = ComputeCost(h, y, weights_array, ...
                                num_data_samples, num_layers, lambda)

%~ cross-entropy cost
cost = (-1/num_data_samples) * sum(sum(y.*log(h) + (1-y).*log(1-h)));

%~ Adding regularization
for layer = 1:num_layers-1
    cost = cost + (weights_array{layer}(:)'*weights_array{layer}(:))...
                   *lambda/(2*num_data_samples);
end

end

