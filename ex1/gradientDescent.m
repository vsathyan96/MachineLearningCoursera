function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

prediction = zeros(m,1);
temp = 0;
temp0 = 0;
temp1 = 0;

for i = 1:m,
prediction(i,1) = theta' * X(i,:)' ;
temp0 = temp0 + (prediction(i,1) - y(i)); 
temp1 = temp1 + ((prediction(i,1) - y(i)) * X(i,2));
end

temp = [temp0; temp1];
temp = temp * alpha /m;
theta = theta - temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
