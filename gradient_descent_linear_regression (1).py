import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

plt.rc('text', usetex = True)
plt.rc('font', family='serif')

from sklearn import linear_model

# ----------------------------------------------------
# plotting parameters
# ----------------------------------------------------
fig_width = 8
fig_height = 6

marker_size = 10

# ----------------------------------------------------
# program
# ----------------------------------------------------
in_file_name = 'linear_regression_test_data.csv'
data_in = pd.read_csv(in_file_name)

x = data_in['x']
y = data_in['y']
x = np.reshape(x, (len(x), 1))
y = np.reshape(y, (len(y), 1))

m = len(x)

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, color='blue')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')
fig.show()

# use sklearn for linear regression
sklearn_linear_regression = linear_model.LinearRegression()
sklearn_linear_regression.fit(x, y)
print sklearn_linear_regression.coef_
print sklearn_linear_regression.intercept_

# To get theta_hat using gradient descent
iteration = 1000

theta = np.zeros((iteration, 2))
theta[0, :] = np.array([2.5, 2.5])

J = np.zeros(iteration)

alpha = 0.1

for index_iter in range(iteration-1):
    partial_derivative = np.array([0.0, 0.0])

    for index_sample in range(m):
        cur_y_hat = theta[index_iter, 0] + theta[index_iter, 1] * x[index_sample]
        cur_residual = cur_y_hat - y[index_sample]

        J[index_iter] = J[index_iter] + cur_residual**2

        partial_derivative[0] = partial_derivative[0] + cur_residual
        partial_derivative[1] = partial_derivative[1] + cur_residual * x[index_sample]

    J[index_iter] = J[index_iter] * 0.5 / m

    theta[index_iter+1, 0] = theta[index_iter, 0] - alpha * partial_derivative[0] / m
    theta[index_iter+1, 1] = theta[index_iter, 1] - alpha * partial_derivative[1] / m

# calculate the last J(theta)
for index_sample in range(m):
    cur_y_hat = theta[iteration-1, 0] + theta[iteration-1, 1] * x[index_sample]
    cur_residual = cur_y_hat - y[index_sample]

    J[iteration-1] = J[iteration-1] + cur_residual**2

J[iteration-1] = J[iteration-1] * 0.5 / m

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$J(\theta)$')
ax.scatter(range(iteration), J, color='blue', s=marker_size)
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$J(\theta)$')
ax.scatter(theta[:, 1], J, color='blue', s=marker_size)
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_0$')
ax.scatter(range(iteration), theta[:, 0], color='blue', s=marker_size)
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_1$')
ax.scatter(range(iteration), theta[:, 1], color='blue', s=marker_size)
fig.show()