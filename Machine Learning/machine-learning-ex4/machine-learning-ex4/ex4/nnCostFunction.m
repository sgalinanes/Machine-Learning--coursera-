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

% Feedforward the neural network, to get our outputs.
layers = 3;
k = num_labels;

H = zeros(m, k);
logs = zeros(m, k);
logs_comp = zeros(m, k);
Y = zeros(m, k);
Z2 = zeros(m, hidden_layer_size);
a1 = zeros(m, input_layer_size+1);
a2 = zeros(m, hidden_layer_size+1);

for i = 1:m
    
    a = X(i, :);

    
    for j = 1:layers-1
        
        a = [1 a];
        a = a';
        
        if(j == 1) 
            theta = Theta1;
            a1(i, :) = a';
        else 
            theta = Theta2;
            a2(i, :) = a';
        end
        
               
        z = theta*a;
        if(j == 1)
            Z2(i, :) = z';
        end
        a = sigmoid(z);
        
        a = a';
    end
       
    % Add logical y rows to the Y matrix %
    Y(i, y(i)) = 1;
    
    
    % Add hypothesis outputs in rows to the H matrix %
    H(i, :) = a;

    
    % Compute logarithm for each row of H %
    logs(i, :) = arrayfun(@(x) log(x), H(i, :));
    
    % Also for 1-h %
    temp = ones(1, k) - H(i, :);
    logs_comp(i, :) = arrayfun(@(x) log(x), temp );
end


% Backward prograpagate to get J gradients %
d3 = H - Y;
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(Z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

reg = trace(Theta1(:, 2:end) * Theta1(:, 2:end)') + trace(Theta2(:, 2:end) * Theta2(:, 2:end)');
add = (lambda/(2*m)) * reg;

J = (-1/m) * ( trace(logs * Y') + trace( logs_comp * ( ones(m,k)- Y)' ) ) + add;


Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

factor = (lambda/m);
Theta1 = factor * Theta1;
Theta2 = factor * Theta2;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
