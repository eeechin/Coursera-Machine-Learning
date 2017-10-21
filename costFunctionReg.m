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

a=X*theta;
hyp = sigmoid(a);

theta_square = theta.^2;
theta_reg = [theta_square(2:end)];


J = sum((y.*log(hyp))+((1-y).*log(1-hyp)))/(-m)+ (sum(theta_reg)*lambda/(2*m));

pre_grad_noreg = (1/m)*((hyp-y)'*X);

grad_noreg = pre_grad_noreg';

grad_reg = grad_noreg.+(lambda/m.*theta);

grad =[grad_noreg(1);grad_reg(2:end)];


% =============================================================

end
