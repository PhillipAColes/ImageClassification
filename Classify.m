clear all;  clc
cd C:\Users\Phillip\Workspace\ML\ImageClassification

% fprintf('Loading data ...\n');
% load('MNIST_handwritten_digits.mat');
% fprintf('... done\n');

imgFile = '.\MNIST\handwritten_digits\train-images-idx3-ubyte';
labelFile = '.\MNIST\handwritten_digits\train-labels-idx1-ubyte';
[imgs labels] = readMNIST(imgFile, labelFile, 60000, 0);
X = reshape(imgs,[400,60000])';
y = labels;

%%

%resize - reshape image
idex = ceil(rand(1)*size(X,1));
y(idex)
digit_image = reshape(X(idex,:),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));

%%
% clear all; clc; clf

%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%%% user should modify the below %%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

% number of classes
num_classes = 10;

% number of hidden layers
num_hidden_layers = 1;

% number of units in each hidden layer, excluding bias
num_hidden_units = [10];

% activation function types
activation_function_type = {'sigmoid', 'sigmoid'};

feature_scaling_tf = false;

% regularisation parameter
lambda = 0.3

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

% if taken from MNIST then we replace 0's with 10's to make indexing work
y(y==0)=10;

% Copy vector of labels from y to yval
yval = y;

% change format of y to ( num_data_samples x num_classes ) binary array
y = zeros(size(y,1),num_classes); 
index = (yval-1).*size(y,1)+(1:size(y,1))';
y(index) = 1;

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
                


options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(backPropagation, nn_weights, options);

%options = optimoptions('fminunc','GradObj','on','Display','iter','MaxIter',200)
%[nn_params, cost] = fminunc(backPropagation, nn_weights, options);

% reshape nn weights
weights_array = Vec2CellArray(nn_params,num_layers,num_units);

%% 

% load and format MNIST test data
imgFile = '.\MNIST\handwritten_digits\t10k-images-idx3-ubyte';
labelFile = '.\MNIST\handwritten_digits\t10k-labels-idx1-ubyte';
[testImgs testLabels] = readMNIST(imgFile, labelFile, 10000, 0);
Xtest = reshape(testImgs,[400,10000])';
ytest = testLabels;
Xtest = [ones(size(Xtest,1),1) Xtest];

%%

% Plot random example from test data
sample = ceil(rand(1)*size(Xtest,1));
digit_image = reshape(Xtest(sample,2:end),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));

% Use our hypothesis to predict which digit is shown in our test data image
activation = ForwardPropagation(weights_array, num_layers,...
                                   1, num_units, ...
                                   activation_function_type, Xtest(sample,:));

% Get digit value (a prediction of 10 = test digit 0)
[predval predidex] = max(activation{num_layers});

% Predicted values of 10 correspond to labels of 0
predidex(predidex==10)=0;

fprintf('Test sample %i has predicted digit: %i,   actual digit: %i\n',...
    sample,predidex,ytest(sample))
