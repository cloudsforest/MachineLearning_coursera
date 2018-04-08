import numpy as np


def computeCost(X,y, theta):
    m = X.shape[0];
    H=X*theta;
    J=np.sum(np.square(H-y))/2/m;
    return J