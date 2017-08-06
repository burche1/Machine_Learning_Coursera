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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%Custo
    %%forward propagation
input_layer = [ones(size(X,1),1) X];  %5000x401
hidden_layer = sigmoid(input_layer*Theta1');     %5000x25
hidden_layer_bias = [ones(size(hidden_layer,1),1) hidden_layer];  %5000x26
output_layer = sigmoid(hidden_layer_bias*Theta2');     %5000x10

tmp_eye = eye(num_labels);
y_ones = tmp_eye(y,:);          %5000x10

J = sum(sum((-log(output_layer).*y_ones)-log(1-output_layer).*(1-y_ones)))*(1/m);

    %%regularização
Theta1_reg = (Theta1).^2;           %25x401
Theta1_without_bias = Theta1_reg(:,2:size(Theta1,2));   %25x400
Theta2_reg = (Theta2.^2);           %10x26
Theta2_without_bias = Theta2_reg(:,2:size(Theta2,2));   %10x25

reg = (lambda/(2*m))*(sum(sum(Theta1_without_bias))+sum(sum(Theta2_without_bias)));

    %%Custo Total
J = J + reg;

    %%backpropagation
delta_output = output_layer - y_ones;       %5000x10
delta_hidden = sigmoidGradient((input_layer*Theta1')).*((delta_output)*(Theta2(:,2:length(Theta2))));    %5000x25

Theta2_grad = (delta_output'*hidden_layer_bias)*(1/m);      %10x26
Theta1_grad = (delta_hidden'*input_layer)*(1/m);            %25x401

    %%backpropagation reg
Theta1_back_reg = (lambda/m)*Theta1(:, 2:size(Theta1, 2));    
Theta1_back_reg = [zeros(size(Theta1, 1), 1) Theta1_back_reg];

Theta2_back_reg = (lambda/m)*Theta2(:, 2:size(Theta2, 2));    
Theta2_back_reg = [zeros(size(Theta2, 1), 1) Theta2_back_reg];

Theta2_grad = Theta2_grad + Theta2_back_reg;
Theta1_grad = Theta1_grad + Theta1_back_reg;
    
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
