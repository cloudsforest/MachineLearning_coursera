import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from warmupExcises import warmupexcises
from computeCost import computeCost
from gradientDescent import gradientDescent
A = warmupexcises();

print('Running warmUpExercise ... \n');
print('5x5 Identity Matrix: \n');
print(A)
fname='ex1data1.txt';
#dtype = np.dtype([('X', 'f8'), ('y', 'f8')])
data = np.loadtxt(fname, delimiter=',')
data=np.asmatrix(data)
y=data[:,1];
m = len(y);

print('Running Gradient Descent ...\n')
X = np.c_[np.ones(m), data[:,0]];
theta = np.zeros((2,1))
J = computeCost(X, y, theta)
print(J)

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;
theta, J = gradientDescent(X, y, theta, alpha, iterations);

#print theta to screen
print('Theta found by gradient descent: ');
print('%f %f \n' % (theta[0], theta[1]));

XX = data[:, 0];
y = data[:, 1];

# #plot using matplotlib
# plt.plot(XX, y,'rs');
# plt.axis([5, 25, -5, 25]);
# plt.hold(True);
# plt.plot(XX, X*theta, '-')
# plt.show()
############
#plot using seaborn


#% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta*10000;
print('For population = 35,000, we predict a profit of %f\n' %predict1);
predict2 = [1, 7] * theta*10000;
print('For population = 70,000, we predict a profit of %f\n' % predict2);



