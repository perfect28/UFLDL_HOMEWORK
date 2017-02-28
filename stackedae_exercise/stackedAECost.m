function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


% 正向传播的向量化实现
m = size(data,2);
aeNumber=numel(stack);%求出稀疏编码的层数
z = cell(aeNumber+1,1);
a = cell(aeNumber+1,1);
a{1} = data;%初始化
for i = 2:aeNumber+1
    z{i} = stack{i-1}.w * a{i-1} + repmat(stack{i-1}.b,1,m);
    a{i} = sigmoid(z{i});
end
% z2 = W1 * data + repmat(b1,1,m);%hiddenSize*visibleSize+visibleSize*m = hiddenSize*m
% a2 = sigmoid(z2);%hiddenSize*m
% z3 = W2 * a2 + repmat(b2,1,m);%visibleSize*hiddenSize+hiddenSize*m = visibleSize*m
% h = sigmoid(z3);%visibleSize*m
   
%-----------------------------------------------------softmax参数的梯度

% M1(r, c) is theta(r)' * x(c)
M1 = softmaxTheta  * a{aeNumber+1};%numClasses*numCases
% preventing overflows
M1 = bsxfun(@minus, M1, max(M1, [], 1));
% M2(r, c) is exp(theta(r)' * x(c))
M2 = exp(M1);
% M2 is the predicted matrix
M2 = bsxfun(@rdivide, M2, sum(M2));
% 1{·} operator only preserve a part of positions of log(M2)
M = groundTruth .* log(M2);%numClasses*numCases

weightDecay =  lambda/2 * sum(sum(softmaxTheta .^2));
cost = -(1/m) * sum(sum(M)) + weightDecay;

% % difference between ground truth and predict value
% diff = groundTruth - M2;%numClasses*numCases
% 
% for i=1:numClasses
%     thetagrad(i,:) = -(1/numCases) * sum((data .* repmat(diff(i,:), inputSize, 1)) ,2)' + lambda * theta(i,:);
%     %numClasses*inputSize的一行
% end

softmaxThetaGrad=-1/m*(groundTruth-M2)*a{aeNumber+1}'+lambda*softmaxTheta;%numClasses*hiddenSize 
%a = hiddenSize*m;

%-----------------------------------------------------自编码层参数梯度 

delta = cell(aeNumber+1);
%delta(l) = theta*(I-P)*f'(z);
delta{aeNumber+1} = softmaxTheta'*(groundTruth-M2).*sigmoidGradient(z{aeNumber+1});
for i = aeNumber:-1:2
    delta{i} = stack{i}.w'*delta{i+1}.*sigmoidGradient(z{i});
end

for i = aeNumber:-1:1
    stackgrad{d}.w = 1/m*delta{i+1}*a{i};
    stackgrad{d}.b = 1/m*sum(delta{i+1},2);
end

% %反向传播的向量化版本
% delta3 = -(data-h).*sigmoidGradient(z3);%visibleSize*m
% delta2 = (W2'*delta3).*sigmoidGradient(z2);%hiddenSize*visibleSize+visibleSize*m=hiddenSize*m
% W2grad = 1/m*delta3*a2'+lambda*W2;%visibleSize*m+m*hiddenSize =  visibleSize*hiddenSize
% W1grad = 1/m*delta2*data'+lambda*W1;%;hiddenSize*m+m*visibleSize = hiddenSize*visibleSize
% b2grad = 1/m*sum(delta3,2);%visibleSize*1
% b1grad = 1/m*sum(delta2,2);%hiddenSize*1
% J = 1/(2*m)*sum(sum((h-data).^2)) + lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2)));
% cost = J;

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end

% You might find this useful
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

