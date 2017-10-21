function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
a=X*theta;

theta_square = theta.^2;
theta_reg = [theta_square(2:end)];
J = sum(power((a.-y),2))/(2*m)+ (sum(theta_reg)*lambda/(2*m));

pre_grad_noreg = (1/m)*((a-y)'*X);
grad_noreg = pre_grad_noreg';
grad_reg = grad_noreg.+(lambda/m.*theta);
grad =[grad_noreg(1);grad_reg(2:end)];

% =========================================================================

grad = grad(:);

end
