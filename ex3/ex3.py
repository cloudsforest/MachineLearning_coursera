import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import seaborn as sbs
import matplotlib.image as mpimg
import scipy.optimize as op

#%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  #% 20x20 Input Images of Digits
num_labels = 10;          #% 10 labels, from 1 to 10
                          #% (note that we have mapped "0" to label 10)
data = loadmat('ex3data1.mat');
def sigmoid(z):
    #%SIGMOID Compute sigmoid function
    #%g = SIGMOID(z) computes the sigmoid of z.

    #% You need to return the following variables correctly
    g = np.matrix(np.zeros(np.shape(z)));

#% ====================== YOUR CODE HERE ======================
#% Instructions: Compute the sigmoid of each value of z (z can be a matrix,vector or scalar).
    g = np.divide(1,(1+np.exp(-z)));
    return g

def lrCostFunction(theta, X, y, lamb):
    #%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    #%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    #%   theta as the parameter for regularized logistic regression and the
    #%   gradient of the cost w.r.t. to the parameters.
    #% Initialize some useful values
    m=len(y);
    J = 0;
    grad = np.zeros(theta.shape);
    n = theta.shape[0];
    if len(np.shape(theta))==1:
        theta = np.transpose(np.matrix(theta));
    #Compute the cost of a particular choice of theta.
    #You should set J to the cost.
    #Compute the partial derivatives and set grad to the partial
    #derivatives of the cost w.r.t. each parameter in theta
    H=sigmoid(X*theta);
    #J = sum((-y.*log(H)-(1-y).*log(1-H)))/m+lambda/2/m*sum(theta(2:n).^2);
    J = np.sum(np.multiply(-y,np.log(H))-np.multiply((1-y),np.log(1-H)))/m \
        +np.sum(np.power(theta[1:n],2))*lamb/2.0/m;
    grad=np.transpose(X)*(H-y)/m+lamb/m*theta;
    grad[0] = grad[0]-lamb/m*theta[0];
    #grad=1
    return J,grad

X=data['X']
y=data['y']
m,n = X.shape
lamb = 0.1;
#% You need to return the following variables correctly
all_theta = np.zeros((num_labels, n + 1));

#% Add ones to the X data matrix
X = np.c_[np.ones(m), X];
optimal_theta = np.zeros((num_labels,n+1))
initial_theta = np.zeros((n + 1, 1));
         #% Set options for fminunc
        #options = optimset('GradObj', 'on', 'MaxIter', 50);
yy = np.matrix(y==1)*1
Result = op.minimize(fun = lrCostFunction,
                    x0 = initial_theta,
                    args = (X, yy,lamb),
                    method = 'TNC',
                    jac = True);
print Result.x