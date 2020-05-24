clear all;  clc
cd C:\Users\Phillip\Workspace\ML\ImageClassification

fprintf('Loading data ...\n');
load('MNIST_handwritten_digits.mat');
fprintf('... done\n');

yval = y;

% number of classes
num_classes = 10;

% change format of y to binary array
y = zeros(size(X,1),num_classes); 
index = (yval-1).*size(X,1)+(1:size(X,1))';
y(index) = 1

%resize - reshape image
idex = 3000%ceil(rand(1)*size(X,1));
yval(idex)
digit_image = reshape(X(idex,:),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));


%%
% clear all; clc; clf

%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%%% user should modify the below %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

% number of hidden layers
num_hidden_layers = 1;

% number of units in each hidden layer, excluding bias
num_hidden_units = [10];

% activation function types
activation_function_type = {'sigmoid', 'sigmoid'};

feature_scaling_tf = false;

% regularisation parameter
lambda = 0.003

%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%

%%% simple test cases ~%%
%load('Test_5.mat')

%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

% number of data samples
num_data_samples = size(y,1);

% number of units in each layer
num_units = [ size(X,2) num_hidden_units num_classes ];

% total number of layers
num_layers = num_hidden_layers + 2;

% preliminary checks for user input parameters
if size(num_hidden_units,2) ~= num_hidden_layers
    error(['Error: array containing number of hidden units in each layer'...
    ' needs to reflect number of hidden layers']);
elseif size(activation_function_type,2) ~= num_hidden_layers+1;
     error(['Error: need to define exactly ',num2str(num_hidden_layers+1),...
    ' activation functions']);       
end

% If weigts array has not been pre-loaded then initialise it
if exist('weights_array')==0
    % constant used for initialising weights
    epsilon_init = 0.12;
    % Initialise weights array
    for layer = 1:num_hidden_layers+1
    
        % num_units(layer)+1 <=== +1 comes from bias term 
        weights_array(layer) = ...
        {rand(num_units(layer)+1,num_units(layer+1)) * 2 * epsilon_init - epsilon_init};
        
    end
end

if feature_scaling_tf == true
    
    X_scaled = ScaleFeatures(X);
    X = X_scaled;
    
end

% Add bias term
X = [ones(num_data_samples,1) X];

%%
%%~~~~~~~~~~~~~~~~~~~~~~%%
%%% Forward propagation %%
%%~~~~~~~~~~~~~~~~~~~~~~%%

[activation] = ForwardPropagation(weights_array, num_layers, num_data_samples, num_units, ...
                                   activation_function_type, X);
        
% our hypothesis, h
h = activation{num_layers};

activation_error{num_layers-1} = h' - y';

cost = ComputeCost(h, y, weights_array, ...
                                num_data_samples, num_layers, lambda)

%%
% Unroll weights ready for backpropagation
nn_weights = [];               
for i=1:num_layers-1
    nn_weights = [nn_weights ; weights_array{i}(:)];
end

reshaped_weights = Vec2CellArray(nn_weights,num_layers,num_units);

[cost unrolled_grad] = NNBackPropagation(nn_weights, X, y, num_layers, ...
                                num_data_samples, num_units, ...
                                activation_function_type, lambda);

grad = Vec2CellArray(unrolled_grad,num_layers,num_units);

% Check numerical gradient                            
%numerical_grad = CalcNumericalGradient(weights_array, num_layers,...
%                    num_data_samples, num_units, ...
%                    activation_function_type, X, y, lambda); 
                
%%

% Create shorthand for cost function to be minimised
backPropagation = @(p) NNBackPropagation(p, X, y, num_layers, ...
                                num_data_samples, num_units, ...
                                activation_function_type, lambda);
                


options = optimset('MaxIter', 200);
[nn_params, cost] = fmincg(backPropagation, nn_weights, options);

%options = optimoptions('fminunc','GradObj','on','Display','iter','MaxIter',200)
%[nn_params, cost] = fminunc(backPropagation, nn_weights, options);

%% 

sample = 1234;

digit_image = reshape(X(sample,2:end),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));

% One final pass of forward propagation to get our final hypothesis
weights_array = Vec2CellArray(nn_params,num_layers,num_units);

activation = ForwardPropagation(weights_array, num_layers,...
                                   1, num_units, ...
                                   activation_function_type, X(sample,:));

[predval predidex] = max(activation{num_layers});

predidex
