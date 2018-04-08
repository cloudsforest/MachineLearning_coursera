import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from pandas import Series, DataFrame
import pandas as pd
import seaborn as sbs
import scipy.optimize as op

fname='ex2data1.txt';
#dtype = np.dtype([('X', 'f8'), ('y', 'f8')])
df = pd.read_csv(fname,delimiter=',',header=None)
df.columns = ['exam1', 'exam2','admitted']
data=np.asmatrix(df)
X = np.matrix(data[:, 0:2]);
y = np.matrix(data[:, 2]);
[m, n] = X.shape;

print(['Plotting data with + indicating (y = 1) examples and o  indicating (y = 0) examples.\n']);
sbs.set_context("notebook", font_scale=1.1)
sbs.set_style("ticks")
sbs.lmplot('exam1', 'exam2',
           data=df,
           fit_reg=False,
           hue="admitted",
        markers=["o", "+"],
          scatter_kws={"marker": "D",
                        "s":50})
plt.title('admission')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

def sigmoid(z):
    #%SIGMOID Compute sigmoid function
    #%g = SIGMOID(z) computes the sigmoid of z.

    #% You need to return the following variables correctly
    g = np.matrix(np.zeros(np.shape(z)));

#% ====================== YOUR CODE HERE ======================
#% Instructions: Compute the sigmoid of each value of z (z can be a matrix,vector or scalar).
    g = np.divide(1,(1+np.exp(-z)));
    return g

def costFunction(theta, X, y):
#%COSTFUNCTION Compute cost and gradient for logistic regression
#%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
#%   parameter for logistic regression and the gradient of the cost
#%   w.r.t. to the parameters.

#% Initialize some useful values
    m = len(y); #% number of training examples

#% You need to return the following variables correctly
    J = 0;
    grad = np.matrix(np.zeros(np.shape(theta)));
    size = np.shape(theta);
    if len(size)==1:
        theta = np.transpose(np.matrix(theta));
#% ====================== YOUR CODE HERE ======================
#% Instructions: Compute the cost of a particular choice of theta.
#%               You should set J to the cost.
#%               Compute the partial derivatives and set grad to the partial
#%               derivatives of the cost w.r.t. each parameter in theta
#%
#% Note: grad should have the same dimensions as theta

    H=sigmoid(X*theta);
    J = np.sum(np.multiply(-y,np.log(H))-np.multiply((1-y),np.log(1-H)))/m;
    grad=np.transpose(X)*(H-y)/m;
    #grad=1
    return J,grad;

#%  Setup the data matrix appropriately, and add ones for the intercept term
#% Add intercept term to x and X_test
X = np.c_[np.ones(m), data[:,0:2]];

#% Initialize fitting parameters
initial_theta = np.matrix(np.zeros((n + 1, 1)));

#% Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y);

print('Cost at initial theta (zeros): %f'% cost);
print('Expected cost (approx): 0.693');
print('Gradient at initial theta (zeros): ');
for i in range(len(grad)): print 'grad=%.4f  ' %(grad[i])
#print(' %f \n' % grad);
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

test_theta = np.transpose(np.matrix([-24,0.2,0.2]));
cost, grad = costFunction(test_theta, X, y);
print('\nCost at test theta: %f'%cost);
print('Expected cost (approx): 0.218');
print('Gradient at test theta: ');
for i in range(len(grad)): print 'grad=%.4f  ' %(grad[i])
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

Result = op.minimize(fun = costFunction,
                    x0 = initial_theta,
                    args = (X, y),
                    method = 'TNC',
                    jac = True);
optimal_theta = Result.x;
cost, grad = costFunction(optimal_theta, X, y);

#% Print theta to screen
print('Cost at theta found by fminunc: %f', cost);
print('Expected cost (approx): 0.203');
print('theta: ');
for i in range(len(optimal_theta)): print 'grad=%.4f  ' %(optimal_theta[i])
print('Expected theta (approx):');
print(' -25.161\n 0.206\n 0.201\n');

