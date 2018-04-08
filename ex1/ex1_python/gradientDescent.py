import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0];
    J_history = np.zeros(iterations);
    for iter in range(0,iterations-1):
        H=X*theta;
        theta0_temp = theta[0,0]-alpha*np.sum(H-y)/m;
        theta1_temp = theta[1,0]-alpha*np.sum(np.multiply((H-y),X[:,1]))/m;
        theta = np.transpose(np.asmatrix([theta0_temp, theta1_temp]))
        #theta = [theta0_temp, theta1_temp];
        J_history[iter] = computeCost(X, y, theta);
    return theta,J_history
