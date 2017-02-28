function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)

% -------------------- YOUR CODE HERE --------------------                                    
% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 

% W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% 
% % Cost and gradient variables (your code needs to compute these values). 
% % Here, we initialize them to zeros. 
% cost = 0;
% W1grad = zeros(size(W1)); 
% W2grad = zeros(size(W2));
% b1grad = zeros(size(b1)); 
% b2grad = zeros(size(b2));
% 
% % 正向传播的向量化实现
% m = size(data,2);
% z2 = W1 * data + repmat(b1,1,m);%hiddenSize*visibleSize+visibleSize*m = hiddenSize*m
% a2 = sigmoid(z2);%hiddenSize*m
% z3 = W2 * a2 + repmat(b2,1,m);%visibleSize*hiddenSize+hiddenSize*m = visibleSize*m
% h = sigmoid(z3);%visibleSize*m
%    
% % 稀疏惩罚Delta
% rho =  sparsityParam;
% rho_hat = mean(a2,2);%hiddenSize*1,求行和
% 
% %反向传播的向量化版本(加入稀疏性限制)
% delta3 = -(data-h);%visibleSize*m
% sparsity_delta = - rho ./ rho_hat + (1 - rho) ./ (1 - rho_hat);
% delta2 = (W2'*delta3+repmat(beta*sparsity_delta,1,m) ).*sigmoidGradient(z2);%hiddenSize*visibleSize+visibleSize*m=hiddenSize*m
% W2grad = 1/m*delta3*a2'+lambda*W2;%visibleSize*m+m*hiddenSize =  visibleSize*hiddenSize
% W1grad = 1/m*delta2*data'+lambda*W1;%;hiddenSize*m+m*visibleSize = hiddenSize*visibleSize
% b2grad = 1/m*sum(delta3,2);%visibleSize*1
% b1grad = 1/m*sum(delta2,2);%hiddenSize*1
% 
% J = 1/(2*m)*sum(sum((h-data).^2)) + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
% KL = sum(rho * log( rho ./ rho_hat) + (1 - rho)* log( (1 - rho) ./ (1 - rho_hat)));
% cost = J +beta*KL;
% 
% %-------------------------------------------------------------------
% % After computing the cost and gradient, we will convert the gradients back
% % to a vector format (suitable for minFunc).  Specifically, we will unroll
% % your gradient matrices into a vector.
% 
% grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];
% 
% end
% 
% %-------------------------------------------------------------------
% % Here's an implementation of the sigmoid function, which you may find useful
% % in your computation of the costs and the gradients.  This inputs a (row or
% % column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
% 
% function sigm = sigmoid(x)
%     sigm = 1 ./ (1 + exp(-x));
% end
% 
% function g = sigmoidGradient(z)
% %SIGMOIDGRADIENT returns the gradient of the sigmoid function
% %evaluated at z
% %   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
% %   evaluated at z. This should work regardless if z is a matrix or a
% %   vector. In particular, if z is a vector or matrix, you should return
% %   the gradient for each element.
%     g = sigmoid(z).*(1-sigmoid(z));
% end
% 


% W1 is a hiddenSize * visibleSize matrix
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 is a visibleSize * hiddenSize matrix
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 is a hiddenSize * 1 vector
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 is a visible * 1 vector
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

numCases = size(data, 2);

% forward propagation
z2 = W1 * data + repmat(b1, 1, numCases);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, numCases);
a3 = z3;

% error
sqrerror = (data - a3) .* (data - a3);
error = sum(sum(sqrerror)) / (2 * numCases);
% weight decay
wtdecay = (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2))) / 2;
% sparsity
rho = sum(a2, 2) ./ numCases;
divergence = sparsityParam .* log(sparsityParam ./ rho) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - rho));
sparsity = sum(divergence);

cost = error + lambda * wtdecay + beta * sparsity;

% delta3 is a visibleSize * numCases matrix
delta3 = -(data - a3);
% delta2 is a hiddenSize * numCases matrix
sparsityterm = beta * (-sparsityParam ./ rho + (1-sparsityParam) ./ (1-rho));
delta2 = (W2' * delta3 + repmat(sparsityterm, 1, numCases)) .* sigmoiddiff(z2);

W1grad = delta2 * data' ./ numCases + lambda * W1;
b1grad = sum(delta2, 2) ./ numCases;

W2grad = delta3 * a2' ./ numCases + lambda * W2;
b2grad = sum(delta3, 2) ./ numCases;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmdiff = sigmoiddiff(x)

    sigmdiff = sigmoid(x) .* (1 - sigmoid(x));
end
