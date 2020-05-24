function [ cost unrolled_grad ] = NNBackPropagation(nn_weights, X, y, num_layers, ...
                                           num_data_samples, num_units, ...
                                           activation_function_type, lambda)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

weights_array = Vec2CellArray(nn_weights,num_layers,num_units);

activation = ForwardPropagation(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X);
                               
h = activation{num_layers};

output_error = (h' - y');

cost = ComputeCost(h, y, weights_array, ...
                                num_data_samples, num_layers, lambda);

grad = cellfun(@(x) x*0,weights_array,'un',0);

unrolled_grad = [];

% Unvectorised approach
for t = 1:num_data_samples   
    % Find the error on the output layer and last hidden layer, as well as
    % the gradient of the cost function w.r.t the weights of the last two
    % layers
    activation_error{num_layers-1} = output_error(:,t);    
    grad{num_layers-1} = grad{num_layers-1} + ...
                         (activation_error{num_layers-1}(:)*activation{num_layers-1}(t,:))';    
    activation_error{num_layers-2} = weights_array{num_layers-1}*activation_error{num_layers-1}...
                         .*activation{num_layers-1}(t,:)'.*(1 - activation{num_layers-1}(t,:))';                                
    grad{num_layers-2} = grad{num_layers-2} + ...
                         (activation_error{num_layers-2}(2:end)*activation{num_layers-2}(t,:))';

    % If we only have one hidden layer
    if( num_layers == 3 )continue;end;                      
    
    % Find the gradient for the remaining hidden layers
    for layer = num_layers-2:-1:2
    activation_error{layer-1} = weights_array{layer}*activation_error{layer}(2:end)...
                         .*activation{layer}(t,:)'.*(1 - activation{layer}(t,:))';                                
    grad{layer-1} = grad{layer-1} + ...
                         (activation_error{layer-1}(2:end)*activation{layer-1}(t,:))';                    
    end
    
end

% Now add regularisation and scale final gradient by number of data samples
% , unroll the resulting gradients so that they can be passed to fminunc
for layer = 1:num_layers-1   
   grad{layer} = 1/num_data_samples * ( grad{layer} + ...
       [ zeros(1,size(weights_array{layer},2)) ; lambda.*weights_array{layer}(2:end,:) ] );
   unrolled_grad = [ unrolled_grad ; grad{layer}(:) ];
end


end

