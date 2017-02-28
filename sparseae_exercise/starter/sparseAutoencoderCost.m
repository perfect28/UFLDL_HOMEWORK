function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% 正向传播的非向量化实现
% m = size(data,2);
% h = zeros(visibleSize,m);
% for i = 1 : m
%   z2 = W1 * data(:,i) + b1;%hiddenSize*visibleSize+visibleSize*1 = hiddenSize*1
%   a2 = sigmoid(z2);
%   z3 = W2 * a2 + b2; %visibleSize*hiddenSize+hiddenSize*1 = visibleSize*1
%   h(:,i) = sigmoid(z3);
% end;
%反向传播的非向量化版本
% for i=1 : m
%   delta3 = -(data(:,i) - h(:,i)) .* sigmoidGradient(z3(:,i)); %visibleSize*1
%   %delta2 = W2'*delta3(:,i) .* sigmoidGradient(z2(:,i));%hiddenSize*visibleSize+visibleSize*1 = hiddenSize*1
%   delta2 = (W2'*delta3(:,i) + beta*sparsity_delta).* sigmoidGradient(z2(:,i)); 
%   
%   gradW2 = gradW2 + delta3*a2(:,i)';%visibleSize*1+1*hiddenSize=visibleSize*hiddenSize
%   gradW1 = gradW1 + delta2*data(:,i)'; %hiddenSize*1+1*visibleSize=hiddenSize*visibleSize
% end;

% 正向传播的向量化实现
m = size(data,2);
z2 = W1 * data + repmat(b1,1,m);%hiddenSize*visibleSize+visibleSize*m = hiddenSize*m
a2 = sigmoid(z2);%hiddenSize*m
z3 = W2 * a2 + repmat(b2,1,m);%visibleSize*hiddenSize+hiddenSize*m = visibleSize*m
h = sigmoid(z3);%visibleSize*m
   
% 稀疏惩罚Delta
rho =  sparsityParam;
rho_hat = mean(a2,2);%hiddenSize*1,求行和

%反向传播的向量化版本(加入稀疏性限制)
delta3 = -(data-h).*sigmoidGradient(z3);%visibleSize*m
sparsity_delta = - rho ./ rho_hat + (1 - rho) ./ (1 - rho_hat);
delta2 = (W2'*delta3+repmat(beta*sparsity_delta,1,m) ).*sigmoidGradient(z2);%hiddenSize*visibleSize+visibleSize*m=hiddenSize*m
W2grad = 1/m*delta3*a2'+lambda*W2;%visibleSize*m+m*hiddenSize =  visibleSize*hiddenSize
W1grad = 1/m*delta2*data'+lambda*W1;%;hiddenSize*m+m*visibleSize = hiddenSize*visibleSize
b2grad = 1/m*sum(delta3,2);%visibleSize*1
b1grad = 1/m*sum(delta2,2);%hiddenSize*1

J = 1/(2*m)*sum(sum((h-data).^2)) + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
KL = sum(rho * log( rho ./ rho_hat) + (1 - rho)* log( (1 - rho) ./ (1 - rho_hat)));
cost = J +beta*KL;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.
    g = sigmoid(z).*(1-sigmoid(z));
end


