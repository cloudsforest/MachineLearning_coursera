import numpy as np;
from computeCost import computeCost


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # %GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    # %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    # %   taking num_iters gradient steps with learning rate alpha

    # % Initialize some useful values
    m = len(y); #% number of training examples
    J_history = np.zeros((num_iters, 1));
    y=np.reshape(y,(len(y),1));

    for iter in range(num_iters):
    # % ====================== YOUR CODE HERE ======================
    # % Instructions: Perform a single gradient step on the parameter vector
    # %               theta.
    # %
    # % Hint: While debugging, it can be useful to print out the values
    # %       of the cost function (computeCostMulti) and gradient here.
        H=np.dot(X,theta);
        theta0_temp = theta[0,0]-alpha*np.sum(np.multiply((H-y),X[:,0:1]))/m;
        theta1_temp = theta[1,0]-alpha*np.sum(np.multiply((H-y),X[:,1:2]))/m;
        theta2_temp = theta[2,0]-alpha*np.sum(np.multiply((H-y),X[:,2:3]))/m;
        theta = np.transpose(np.asmatrix([theta0_temp,theta1_temp,theta2_temp]));
    # % Save the cost J in every iteration
        J_history[iter] = computeCost(X, y, theta);

    return theta, J_history