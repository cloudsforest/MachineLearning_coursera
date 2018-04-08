clc;
clear;
close all;
data=load('ex1data2.txt');
data_size = size(data);
feature_size = data_size(2); 
y=data(:,feature_size);
m=length(y);
%X = [ones(m, 1), data(:,1), data(:,2)];
X = data(:,1:(feature_size-1));
[X_norm, mu, sigma] = featureNormalize(X);
X_norm =[ones(m, 1), X_norm];
theta = zeros(feature_size,1);
J = computeCostMulti(X_norm, y, theta);
alpha=0.0225;
num_iters=1500;
[theta, J_history] = gradientDescentMulti(X_norm, y, theta, alpha, num_iters);
fprintf('Theta found by gradient descent: ');
fprintf('%f %f %f\n', theta(1), theta(2),theta(3));
figure
plot(J_history)