function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%Forward propagation
%input layer
a1 = [ones(1,m); X'];

%hidden layer 1
z2 = Theta1 * a1;
a2 = [ones(1,m); sigmoid(z2)];

%output layer
z3 = Theta2 * a2;
a3 = sigmoid(z3);


%Expand y so that position of Y[i] = y
Y = eye(num_labels)(:,y);

%Cost function 
J = (1/m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

%Regularization
J += (lambda/ (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2))); 
J += (lambda/ (2 * m)) * (sum(sum(Theta2(:, 2:end) .^ 2)));

%Backpropagation
delta3 =  a3 - Y; %Y not y! debugging nightmare 
delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2);

Theta2_grad = ((1/m) * (delta3 * a2'));
Theta1_grad = ((1/m) * (delta2 * a1'));

Theta2_grad += (lambda / m) * ([zeros(size(Theta2,1) ,1),Theta2(:, 2:end)]);
Theta1_grad += (lambda / m) * ([zeros(size(Theta1,1) ,1),Theta1(:, 2:end)]);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
