import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from normalEqn import normalEqn


print('Loading data ...\n');
# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]; # 0:2 not include 2
y = data[:, 2];
m = np.size(y);

#% Print out some data points
print('First 10 examples from the dataset: \n');
for i in range(len(X[0:10,0])): print 'x=%.0f %.0f, y=%.0f' %(X[i,0],X[i,1],y[i])

#% Scale features and set them to zero mean
print('Normalizing Features ...\n');
X, mu, sigma = featureNormalize(X);
#% Add intercept term to X
X=np.c_[np.ones(m), X];
print('Running gradient descent ...\n');

# % Choose some alpha value
alpha = 0.01;
num_iters = 400;

# % Init Theta and Run Gradient Descent
theta = np.zeros((3, 1));
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters);

#% Plot the convergence graph

plt.plot(range(num_iters), J_history,'-b');
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

#% Display gradient descent's result
print('Theta computed from gradient descent: \n');
for i in range(len(theta)): print 'theta=%.0f ' %(theta[i])
#print(' %f \n', theta);

# % Estimate the price of a 1650 sq-ft, 3 br house
# % ====================== YOUR CODE HERE ======================
# % Recall that the first column of X is all-ones. Thus, it does
# % not need to be normalized.
x_targe=[1650,3];
x_targe=np.insert((x_targe-mu)/sigma,0,1)
x_targe=np.matrix(x_targe);
price =x_targe*theta; # You should change this
print('predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n' %price);
x_targe=[2450,3];
x_targe=np.insert((x_targe-mu)/sigma,0,1)
x_targe=np.matrix(x_targe);
price =x_targe*theta; # You should change this
print('predicted price of a 2450 sq-ft, 3 br house (using gradient descent):\n $%f\n' %price);
x_targe=[1415,2];
x_targe=np.insert((x_targe-mu)/sigma,0,1)
x_targe=np.matrix(x_targe);
price =x_targe*theta; # You should change this
print('predicted price of a 1415 sq-ft, 2 br house (using gradient descent):\n $%f\n' %price);

print('Plotting Training and regressioned results by gradient descent.\n');
X = data[:, 0:2];
X=np.c_[np.ones(m), X];# %denormalize features

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:,0:1], X[:,1:2], y, c='r', marker='o')



fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.gca(projection='3d');
plt.hold(True);
xx =np.linspace(0,5000,25);
yy =np.linspace(1,5,25);
xx_surf,yy_surf=np.meshgrid(xx,yy);
zz = np.zeros((len(xx),len(yy)));
# ax.set_xlim(0, 6000)
# ax.set_ylim(0, 6)
# ax.set_zlim(30000, 800000)
for i,x_sim in enumerate(xx):
    for j,y_sim in enumerate(yy):
         zz[i,j] = [1,(x_sim-mu[0])/sigma[0],(y_sim-mu[1])/sigma[1]]*theta;
ax.plot_surface(xx_surf, yy_surf, zz, rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0, antialiased=True,alpha=0.5);
ax.scatter(X[:,1:2], X[:,2:3], y, c='r', marker='o')
ax.set_xlabel('sq-ft of room')
ax.set_ylabel('#bedroom')
ax.set_zlabel('price')
plt.show();

print('Solving with normal equations...\n');
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]; # 0:2 not include 2
y = data[:, 2];
m = np.size(y);
X=np.c_[np.ones(m), X];

theta = normalEqn(X, y);

#% Display normal equation's result
print('Theta computed from the normal equations: \n');
for i in xrange(len(theta)): print 'theta=%.0f ' %(theta[i])
theta = np.matrix(theta);
print (np.shape(theta))



#% Estimate the price of a 1650 sq-ft, 3 br house
#% ====================== YOUR CODE HERE ======================
x_targe=[1, 1650, 3];
x_targe=np.matrix(x_targe);
print ( np.shape(x_targe))
#price =np.dot(x_targe,theta);
price=theta*np.transpose(x_targe);
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n' % price);
x_targe=[1, 2450, 3];
x_targe=np.matrix(x_targe);
#price =np.dot(x_targe,theta);
price=theta*np.transpose(x_targe);
print('Predicted price of a 2450 sq-ft, 3 br house (using gradient descent):\n $%f\n' % price);
x_targe=[1, 1415, 2];
x_targe=np.matrix(x_targe);
#price =np.dot(x_targe,theta);
price=theta*np.transpose(x_targe);
print('Predicted price of a 1415 sq-ft, 3 br house (using gradient descent):\n $%f\n' % price);











