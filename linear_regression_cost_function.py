import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

plt.rc('text', usetex = True)
plt.rc('font', family='serif')

# ====================================================================
fig_width = 8
fig_height = 6

m = 20
x = 10 *np.random.rand(m)

theta_0 = 0
theta_1 = 2
sigma = 1

epsilon = np.random.normal(loc=0, scale=sigma, size=m)

y = theta_0 + theta_1 * x + epsilon

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.scatter(x, y, color='blue')
ax.set_xlabel(r'$x$', fontsize=16)
ax.set_ylabel(r'$y$', fontsize=16)
fig.show()
out_file_name = 'data.pdf'
fig.savefig(out_file_name, dpi = 300)

# theta_0 = 0 and J is a function of theta_1
theta_0_grid = np.array([0])
theta_1_grid = np.arange(start=0.0, stop=4.0, step=0.1)
# J(theta)
J = np.zeros((len(theta_0_grid), len(theta_1_grid)))
for index_theta_0 in range(len(theta_0_grid)):
    cur_theta_0_parameter = theta_0_grid[index_theta_0]
    for index_theta_1 in range(len(theta_1_grid)):
        cur_theta_1_parameter = theta_1_grid[index_theta_1]
        for index_sample in range(m):
            single_residual = cur_theta_0_parameter + cur_theta_1_parameter * x[index_sample] - y[index_sample]
            J[index_theta_0, index_theta_1] = J[index_theta_0, index_theta_1] + single_residual**2

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
ax.scatter(theta_1_grid, J, color='blue')
ax.plot(theta_1_grid, J[0, :], color='blue')
ax.set_xlabel(r'$\theta_1$', fontsize=16)
ax.set_ylabel(r'$J$', fontsize=16)
fig.show()
out_file_name = 'J of theta_1.pdf'
fig.savefig(out_file_name)

# theta_0 != 0 and J is a function of theta_0 and theta_1
theta_0_grid = np.arange(start=-3.0, stop=5.0, step=0.1)
theta_1_grid = np.arange(start=-3.0, stop=7.0, step=0.1)
# J(theta)
J = np.zeros((len(theta_1_grid), len(theta_0_grid)))
for index_theta_0 in range(len(theta_0_grid)):
    cur_theta_0_parameter = theta_0_grid[index_theta_0]
    for index_theta_1 in range(len(theta_1_grid)):
        cur_theta_1_parameter = theta_1_grid[index_theta_1]
        for index_sample in range(m):
            single_residual = cur_theta_0_parameter + cur_theta_1_parameter * x[index_sample] - y[index_sample]
            J[index_theta_1, index_theta_0] = J[index_theta_1, index_theta_0] + single_residual**2

# plot J as a function of theta_0 and theta_1
theta_0_mesh, theta_1_mesh = np.meshgrid(theta_0_grid, theta_1_grid)

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta_0_mesh, theta_1_mesh, J,
                       cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(r'$\theta_0$', fontsize=16)
ax.set_ylabel(r'$\theta_1$', fontsize=16)
ax.set_zlabel(r'$J$', fontsize=16)
fig.show()
out_file_name = 'J of theta_0 and theta_1.pdf'
fig.savefig(out_file_name)




