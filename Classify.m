% CLASSIFY is the main routine used to classify handwritten digits

clear all;  clc
cd C:\Users\Phillip\Workspace\ML\ImageClassification

%~ load MNIST handwritten digits training set data
imgFile = '.\MNIST\handwritten_digits\train-images-idx3-ubyte';
labelFile = '.\MNIST\handwritten_digits\train-labels-idx1-ubyte';
[imgs labels] = readMNIST(imgFile, labelFile, 60000, 0);
X = reshape(imgs,[400,60000])';
y = labels;

%%
%~ plot random image from our training set 
idex = ceil(rand(1)*size(X,1));
y(idex)
digit_image = reshape(X(idex,:),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));

%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%
%%~~~~~~~~~~~~~~~~~~~  User should modify the below  ~~~~~~~~~~~~~~~~~~~~%%
%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~%%

%~ number of classes
num_classes = 10;

%~ number of hidden layers
num_hidden_layers = 1;

%~ number of units in each hidden layer, excluding bias
num_hidden_units = [10];

%~ activation function types
activation_function_type = {'sigmoid', 'sigmoid'};

feature_scaling_tf = false;

%~ regularisation parameter
lambda = 0.3

%~ turn gradient checking on/off (SLOW!)
grad_check = false;

%~ if we want to load a pre-existing model
%load('Test_5.mat')

%%~~~~~~~~~~~~~~~~~~ End of user modification section ~~~~~~~~~~~~~~~~~~~%%

%%
%~ number of data samples
num_data_samples = size(y,1);

%~ number of units in each layer
num_units = [ size(X,2) num_hidden_units num_classes ];

%~ total number of layers
num_layers = num_hidden_layers + 2;

%~ preliminary checks for user input parameters
if size(num_hidden_units,2) ~= num_hidden_layers
    error(['Error: array containing number of hidden units in each layer'...
    ' needs to reflect number of hidden layers']);
elseif size(activation_function_type,2) ~= num_hidden_layers+1;
     error(['Error: need to define exactly ',num2str(num_hidden_layers+1),...
    ' activation functions']);       
end

%~ If weigts array has not been pre-loaded then initialise it
if exist('weights_array')==0
    %~ constant used for initialising weights
    epsilon_init = 0.12;
    %~ Initialise weights array
    for layer = 1:num_hidden_layers+1
    
        %~ num_units(layer)+1 <=== +1 comes from bias term 
        weights_array(layer) = ...
        {rand(num_units(layer)+1,num_units(layer+1)) * 2 * epsilon_init - epsilon_init};
        
    end
end

if feature_scaling_tf == true
    X_scaled = ScaleFeatures(X);
    X = X_scaled;    
end

%~ Add bias term
X = [ones(num_data_samples,1) X];

%%

%~ if taken from MNIST then we replace 0's with 10's to make indexing work
y(y==0)=10;

%~ Copy vector of labels from y to yval
yval = y;

%~ change format of y to ( num_data_samples x num_classes ) binary array
y = zeros(size(y,1),num_classes); 
index = (yval-1).*size(y,1)+(1:size(y,1))';
y(index) = 1;

%%
%~ perform gradient checking by comparing gradients calculated from
%~ backprop with gradients estimated by finite differences method
if grad_check == true

    nn_weights = [];               
    for i=1:num_layers-1
        nn_weights = [nn_weights ; weights_array{i}(:)];
    end
    
    %~ perform one pass of backprop to get the gradients
    [cost unrolled_grad] = NNBackPropagation(nn_weights, X, y, num_layers, ...
                                num_data_samples, num_units, ...
                                activation_function_type, lambda);

    grad = Vec2CellArray(unrolled_grad,num_layers,num_units);

    %~ Check numerical gradient... SLOW!!!                           
    numerical_grad = CalcNumericalGradient(weights_array, num_layers,...
                    num_data_samples, num_units, ...
                    activation_function_type, X, y, lambda); 
    
    grad_diff = cellfun(@minus,grad,numerical_grad,'Un',0);
    
    %~ if the difference between backprop and finite diffs gradients 
    %~ exceeds 1e-8 then print a warning
    for i=1:num_layers-1
        max_diff = max(max(grad_diff{i}));
        if abs(max_diff) > 1e-8
            fprintf('WARNING: backprop gradient is not equal to finite diff',...
                    ' gradient! Continuing.')
        end
    end
    
end

%%
%~ Unroll weights ready for backpropagation routine, as fminunc and fmincg
%~ require them in this format
nn_weights = [];               
for i=1:num_layers-1
    nn_weights = [nn_weights ; weights_array{i}(:)];
end

%~ Create shorthand for cost function to be minimised
backPropagation = @(p) NNBackPropagation(p, X, y, num_layers, ...
                                num_data_samples, num_units, ...
                                activation_function_type, lambda);
                
options = optimset('MaxIter', 50);
[nn_params, cost] = fmincg(backPropagation, nn_weights, options);

%~ fminunc finds the minimum of an unconstrained multivariable function.
%~ Here we use the gradient determined from our backprop function
%options = optimoptions('fminunc','GradObj','on','Display','iter','MaxIter',200)
%[nn_params, cost] = fminunc(backPropagation, nn_weights, options);

%~ reshape the nn weights to use later in Forward Prop
weights_array = Vec2CellArray(nn_params,num_layers,num_units);

%% 

%~ load and format MNIST test data
imgFile = '.\MNIST\handwritten_digits\t10k-images-idx3-ubyte';
labelFile = '.\MNIST\handwritten_digits\t10k-labels-idx1-ubyte';
[testImgs testLabels] = readMNIST(imgFile, labelFile, 10000, 0);
Xtest = reshape(testImgs,[400,10000])';
ytest = testLabels;
Xtest = [ones(size(Xtest,1),1) Xtest];
num_test_samples = size(Xtest,1);

%%

%~ Use our hypothesis to predict which digit is shown in our test data image
activation = ForwardPropagation(weights_array, num_layers,...
                                   num_test_samples, num_units, ...
                                   activation_function_type, Xtest);

%~ Get digit value (a prediction of 10 = test digit 0)
[predval predidex] = max(activation{num_layers},[],2);

%~ Predicted values of 10 correspond to labels of 0
predidex(predidex==10)=0;

%~ Accuracy of model
accuracy = 1 - sum(predidex~=ytest)/num_test_samples;
fprintf('Accuracy of model is %f%%\n',accuracy*100)

%~ Plot random example from test data
sample = ceil(rand(1)*size(Xtest,1));
digit_image = reshape(Xtest(sample,2:end),[20,20]);
imagesc(digit_image);
colormap(flipud(gray));
fprintf('Random test sample %i has predicted digit: %i,   actual digit: %i\n',...
    sample,predidex(sample),ytest(sample))
