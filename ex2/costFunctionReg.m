function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
reg = 0;
[l p] = size(theta);

for i = 1:m,
J = J + ([y(i)*log(sigmoid(theta' * X(i,:)'))] + [(1-y(i))*log(1-sigmoid(theta' * X(i,:)'))]);
end;
J = -J/m;

for j = 1:l,
for i = 1:m,
grad(j) = grad(j) + [(sigmoid(theta' * X(i,:)') - y(i)) * X(i,j)];
end;
end;
grad = grad/m;

for j = 1:l,
theta(1,p) = 0; % Exclude feature bias
reg = reg + (theta(j,p) * theta(j,p));
grad(j) = grad(j) + (lambda * theta(j,p)/m);
end;
reg = ((lambda * reg )/(2*m));
J = J + reg;

% =============================================================

end
