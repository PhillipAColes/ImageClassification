% NNBACKPROPAGATION computes the gradient of the cost wrt the weights using
% backpropagation.
% - Parameters -
% nn_weights: unrolled array of weights for the neural network
% X: features
% y: labels 
% num_layers: numer of layers of the neural network
% num_data_samples: number of data samples
% num_units: array containing numer of units in each layer of network
% activation_function_type: array containing strings that define the
%                           activation functions of each layer
% lambda: regularisation parameter
% - Returns -
% cost: cost as calculated by function ComputeCost
% unrolled_grad: unrolled (one dimensional array of) gradients  

function [ cost unrolled_grad ] = NNBackPropagation(nn_weights, X, y, num_layers, ...
                                           num_data_samples, num_units, ...
                                           activation_function_type, lambda)

weights_array = Vec2CellArray(nn_weights,num_layers,num_units);

activation = ForwardPropagation(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X);
                               
h = activation{num_layers};

cost = ComputeCost(h, y, weights_array, ...
                                num_data_samples, num_layers, lambda);

output_error = (h' - y');

%~ initialise cell array to hold gradient values from weights array
grad = cellfun(@(x) x*0,weights_array,'un',0);

unrolled_grad = [];

%~ Slower unvectorised approach for the moment...
for t = 1:num_data_samples   
    %~ Find the error on the output layer and last hidden layer, as well as
    %~ the gradient of the cost function w.r.t the weights of the last two
    %~ layers
    activation_error{num_layers-1} = output_error(:,t);    
    grad{num_layers-1} = grad{num_layers-1} + ...
                         (activation_error{num_layers-1}(:)*activation{num_layers-1}(t,:))';    
    activation_error{num_layers-2} = weights_array{num_layers-1}*activation_error{num_layers-1}...
                         .*activation{num_layers-1}(t,:)'.*(1 - activation{num_layers-1}(t,:))';                                
    grad{num_layers-2} = grad{num_layers-2} + ...
                         (activation_error{num_layers-2}(2:end)*activation{num_layers-2}(t,:))';

    %~ If we only have one hidden layer
    if( num_layers == 3 )continue;end;                      
    
    %~ Find the gradient for the remaining hidden layers
    for layer = num_layers-2:-1:2
    activation_error{layer-1} = weights_array{layer}*activation_error{layer}(2:end)...
                         .*activation{layer}(t,:)'.*(1 - activation{layer}(t,:))';                                
    grad{layer-1} = grad{layer-1} + ...
                         (activation_error{layer-1}(2:end)*activation{layer-1}(t,:))';                    
    end
    
end

%~ Now add regularisation and scale final gradient by number of data samples
%~ , unroll the resulting gradients so that they can be passed to fminunc/cg
for layer = 1:num_layers-1   
   grad{layer} = 1/num_data_samples * ( grad{layer} + ...
       [ zeros(1,size(weights_array{layer},2)) ; lambda.*weights_array{layer}(2:end,:) ] );
   unrolled_grad = [ unrolled_grad ; grad{layer}(:) ];
end


end

