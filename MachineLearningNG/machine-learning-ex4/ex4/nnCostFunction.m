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

%a(1)0
one1 = ones(m,1);
X_1=[one1 X];   
% Theta1 hidden_layer_size, (input_layer_size + 1)   X  m,(input_layer_size + 1)
z2m= Theta1 * X_1'; 
a2m= sigmoid(z2m);   % hidden_layer_size,m

%a(2)0
one2= ones(1,m);
a2m=[one2;a2m];  %hidden_layer_size+1,m

z3m= Theta2 * a2m; % Theta2 num_labels, (hidden_layer_size + 1);  

h = sigmoid(z3m); % num_labels,m
h=h';  % m,num_labels

for i=1:m
  yi=zeros(1,num_labels);
  yi(y(i)) = 1;
  J=J+(yi*log(h(i,:)')+(1-yi)*log(1-h(i,:)'));
end


J= J*(-1)/m;

J=J+(sum(Theta1(:,2:end)(:).^2)+sum(Theta2(:,2:end)(:).^2))*lambda/(2*m);



for i=1:m
  a1=X(i,:)';     %input_layer_size,1
  a1=[1; a1];%(input_layer_size + 1),1
  z2=Theta1*a1; % hidden_layer_size ,1
  a2=sigmoid(z2);% hidden_layer_size ,1
  a2=[1;a2];    %hidden_layer_size+1 ,1
  z3=Theta2*a2; %num_labels,1
  a3=sigmoid(z3); %num_labels,1
  yi=zeros(num_labels,1);
  yi(y(i))=1;
  fprintf("y(i) %f",yi);
  sigma3 = a3-yi;%num_labels,1
  delta=Theta2'*sigma3;
  delta=delta(2:end);
  sigma2 = delta.*sigmoidGradient(z2); %hidden_layer_size , 1
  Theta1_grad=Theta1_grad+sigma2*a1';  %hidden_layer_size,input_layer_size+1
  Theta2_grad=Theta2_grad+sigma3*a2';  %num_labels,hidden_layer_size+1
end


beta1 = (lambda/m)*Theta1;
beta1(:,1)=0;
Theta1_grad= Theta1_grad./m + beta1;

beta2 = (lambda/m)*Theta2;
beta2(:,1)=0;
Theta2_grad=Theta2_grad./m + beta2;
  


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
