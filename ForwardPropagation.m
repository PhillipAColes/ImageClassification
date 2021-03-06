%FORWARDPROPAGATION...

function [activation] = ForwardPropagation(weights_array, num_layers,...
                                   num_data_samples, num_units, ...
                                   activation_function_type, X)

activation(1) = {X};

for layer = 1:num_layers-1

    if strcmp(activation_function_type{layer},'sigmoid')
        z = activation{layer}*weights_array{layer};
        activation_next_layer = 1 ./ (1+exp(-z));
    elseif strcmp(activation_function_type{layer},'tanh')
        z = activation{layer}*weights_array{layer};
        activation_next_layer = tanh(z);
    elseif strcmp(activation_function_type{layer},'linear')
        activation_next_layer = activation{layer}*weights_array{layer};
    end
    
    % Fill cell of node activation arrays using forward propogation
    if layer+1 == num_layers
        % No bias term added for output layer
        activation(layer+1) = {activation_next_layer};
    else
       % Array of ones for bias term
       activation(layer+1) = {[ones(num_data_samples,1) activation_next_layer]};
    end
  
end


end

                             