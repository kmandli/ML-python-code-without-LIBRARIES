import numpy as np
import math
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

# for using latex
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
plt.rc('text', usetex = True)
plt.rc('font', family='serif')

from ML_toolbox import d_LDA

# -----------------------------------------------------
# plotting parameters
# -----------------------------------------------------
fig_width = 8
fig_height = 6

line_width = 2
marker_size = 7

axis_label_font_size = 9
legend_font_size = 9

# my customization of plotting
plot_params = {'figure.figsize': (fig_width, fig_height)}
plt.rcParams.update(plot_params)

# -----------------------------------------------------
# 1. Do LDA on toy data
# -----------------------------------------------------
x1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
x2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

# LDA
W, mu_1_tilde, mu_2_tilde = d_LDA.d_LDA_two_class(x1, x2)
W_scaled = W * 12.0 / W[0]

# projections
projection_1 = np.matmul(W, x1.transpose())
projection_2 = np.matmul(W, x2.transpose())

# slope of W
theta = math.atan(W[1] / W[0])

# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_title('Apply LDA to a toy dataset')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')

ax.scatter(x1[:, 0], x1[:, 1], color='blue')
ax.scatter(x2[:, 0], x2[:, 1], color='red')

ax.plot([0, W_scaled[0]], [0, W_scaled[1]], color='green')
ax.plot(-np.array([projection_1])*math.cos(theta), -np.array([projection_1])*math.sin(theta), color='blue', marker='x', markersize=marker_size)
ax.plot(-np.array([projection_2])*math.cos(theta), -np.array([projection_2])*math.sin(theta), color='red', marker='x', markersize=marker_size)

ax.set_aspect(1)

fig.show()

# -----------------------------------------------------
# 2. Do LDA on iris data set: setosa and virginica
# -----------------------------------------------------
iris = load_iris()

II_setosa = np.where(iris.target==0)
II_versicolor = np.where(iris.target==1)
II_virginica = np.where(iris.target==2)

II_setosa = II_setosa[0]
II_versicolor = II_versicolor[0]
II_virginica = II_virginica[0]

# apply LDA to setosa and virginica data
fig = plt.figure()
fig.suptitle('iris data: setosa and virginica')

ax = fig.add_subplot(1, 3, 1)
ax.plot(iris.data[II_setosa, 0], iris.data[II_setosa, 1], linestyle='None', marker='o', markersize=marker_size, color='blue', label='setosa')
ax.plot(iris.data[II_virginica, 0], iris.data[II_virginica, 1], linestyle='None', marker='o', markersize=marker_size, color='red', label='virginica')
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.legend()


ax = fig.add_subplot(1, 3, 2)
ax.plot(iris.data[II_setosa, 2], iris.data[II_setosa, 3], linestyle='None', marker='o', markersize=marker_size, color='blue')
ax.plot(iris.data[II_virginica, 2], iris.data[II_virginica, 3], linestyle='None', marker='o', markersize=marker_size, color='red')
ax.set_xlabel('petal length')
ax.set_ylabel('petal width')

ax = fig.add_subplot(1, 3, 3)
ax.plot(iris.data[II_setosa, 1], iris.data[II_setosa, 2], linestyle='None', marker='o', markersize=marker_size, color='blue')
ax.plot(iris.data[II_virginica, 1], iris.data[II_virginica, 2], linestyle='None', marker='o', markersize=marker_size, color='red')
ax.set_xlabel('sepal width')
ax.set_ylabel('petal length')

fig.show()

x1 = iris.data[II_setosa, :]
x2 = iris.data[II_virginica, :]

W, mu_1_tilde, mu_2_tilde = d_LDA.d_LDA_two_class(x1, x2)

# projection x onto W
projection_1 = np.matmul(W, x1.transpose())
projection_2 = np.matmul(W, x2.transpose())

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA to setosa and virginica data')
ax.set_xlabel('projection onto W')
ax.set_ylabel('')
ax.plot(projection_1, np.zeros(len(projection_1)), linestyle='None', marker='o', markersize=marker_size, color='blue')
ax.plot(projection_2, np.zeros(len(projection_2)), linestyle='None', marker='o', markersize=marker_size, color='red')
ax.plot(mu_1_tilde, 0.0, linestyle='None', marker='*', markersize=20, color='magenta')
ax.plot(mu_2_tilde, 0.0, linestyle='None', marker='*', markersize=20, color='magenta')
fig.show()

# -----------------------------------------------------
# 3. Do LDA on iris data set: versicolor and virginica
# -----------------------------------------------------
fig = plt.figure()
fig.suptitle('iris data: versicolor and virginica')
ax = fig.add_subplot(1, 3, 1)
ax.plot(iris.data[II_versicolor, 0], iris.data[II_versicolor, 1], linestyle='None', marker='o', markersize=marker_size, color='blue', label='versicolor')
ax.plot(iris.data[II_virginica, 0], iris.data[II_virginica, 1], linestyle='None', marker='o', markersize=marker_size, color='red', label='virginica')
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
ax.legend()

ax = fig.add_subplot(1, 3, 2)
ax.plot(iris.data[II_versicolor, 2], iris.data[II_versicolor, 3], linestyle='None', marker='o', markersize=marker_size, color='blue')
ax.plot(iris.data[II_virginica, 2], iris.data[II_virginica, 3], linestyle='None', marker='o', markersize=marker_size, color='red')
ax.set_xlabel('petal length')
ax.set_ylabel('petal width')

ax = fig.add_subplot(1, 3, 3)
ax.plot(iris.data[II_versicolor, 1], iris.data[II_versicolor, 2], linestyle='None', marker='o', markersize=marker_size, color='blue')
ax.plot(iris.data[II_virginica, 1], iris.data[II_virginica, 2], linestyle='None', marker='o', markersize=marker_size, color='red')
ax.set_xlabel('sepal width')
ax.set_ylabel('petal length')

fig.show()

x1 = iris.data[II_versicolor, :]
x2 = iris.data[II_virginica, :]

W, mu_1_tilde, mu_2_tilde = d_LDA.d_LDA_two_class(x1, x2)

# projection x onto W
projection_1 = np.matmul(W, x1.transpose())
projection_2 = np.matmul(W, x2.transpose())

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA to versicolor and virginica data')
ax.set_xlabel('projection onto W')
ax.set_ylabel('')
ax.plot(projection_1, np.zeros(len(projection_1)), linestyle='None', marker='o', markersize=marker_size, color='blue', label='versicolor')
ax.plot(projection_2, np.zeros(len(projection_2)), linestyle='None', marker='o', markersize=marker_size, color='red', label='virginica')
ax.plot(mu_1_tilde, 0.0, linestyle='None', marker='*', markersize=20, color='magenta')
ax.plot(mu_2_tilde, 0.0, linestyle='None', marker='*', markersize=20, color='magenta')
ax.legend()

fig.show()

# -----------------------------------------------------
# 4. Do LDA on iris data set: setosa, versicolor, and virginica
# -----------------------------------------------------
W, eigenvalues, mu_tilde_dict = d_LDA.d_LDA_multi_class(iris)

# project iris data onto W
projection = np.matmul(W.transpose(), iris.data.transpose())
projection = projection.transpose()

# plot the projections
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA to iris')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel(r'$W_2$')
ax.plot(projection[0:50, 0], projection[0:50, 1], linestyle='None', marker='o', markersize=marker_size, color='blue', label='setosa')
ax.plot(projection[50:100, 0], projection[50:100, 1], linestyle='None', marker='o', markersize=marker_size, color='red', label='versicolor')
ax.plot(projection[100:150, 0], projection[100:150, 1], linestyle='None', marker='o', markersize=marker_size, color='green', label='setosa')
ax.legend()

for i in range(len(mu_tilde_dict.keys())):
    ax.plot(mu_tilde_dict[i][0], mu_tilde_dict[i][1],
            linestyle='None', marker='*', markersize=15, color='magenta')

fig.show()

# -----------------------------------------------------
# 5. Do LDA on cell line data
# -----------------------------------------------------
in_file_name = "SCLC_study_output_filtered_2.csv"
data_in = pd.read_csv(in_file_name, index_col=0)
X = data_in.as_matrix()
y = np.concatenate((np.zeros(20), np.ones(20)))

II_0 = np.where(y==0)
II_1 = np.where(y==1)

II_0 = II_0[0]
II_1 = II_1[0]

W, mu_1_tilde, mu_2_tilde = d_LDA.d_LDA_two_class(X[II_0, :], X[II_1, :])

projection = np.matmul(X, W)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying LDA to cell line data')
ax.set_xlabel('projection')
ax.set_ylabel('')
ax.plot(projection[0:20], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
ax.plot(projection[20:40], np.zeros(20), linestyle='None', marker='o', markersize=marker_size, color='red', label='NSCLC')
ax.plot(mu_1_tilde, 0.0, linestyle='None', marker='*', markersize=15, color='magenta')
ax.plot(mu_2_tilde, 0.0, linestyle='None', marker='*', markersize=15, color='magenta')
ax.legend()

fig.show()

# -----------------------------------------------------
# 6. use sklearn LDA
# -----------------------------------------------------
# apply sklearn LDA to iris data
sklearn_LDA = LDA(n_components=2)
sklearn_LDA_projection = sklearn_LDA.fit_transform(iris.data, iris.target)
sklearn_LDA_projection = -sklearn_LDA_projection

# plot the projections
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying sklearn LDA to iris')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel(r'$W_2$')
ax.plot(sklearn_LDA_projection[0:50, 0], sklearn_LDA_projection[0:50, 1], linestyle='None', marker='o', markersize=marker_size, color='blue', label='setosa')
ax.plot(sklearn_LDA_projection[50:100, 0], sklearn_LDA_projection[50:100, 1], linestyle='None', marker='o', markersize=marker_size, color='red', label='versicolor')
ax.plot(sklearn_LDA_projection[100:150, 0], sklearn_LDA_projection[100:150, 1], linestyle='None', marker='o', markersize=marker_size, color='green', label='setosa')
ax.legend()

fig.show()

# apply sklearn LDA to cell line data
sklearn_LDA = LDA(n_components=2)
sklearn_LDA_projection = sklearn_LDA.fit_transform(X, y)
sklearn_LDA_projection = -sklearn_LDA_projection

# plot the projections
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_title('Results from applying sklearn LDA to cell line data')
ax.set_xlabel(r'$W_1$')
ax.set_ylabel('')
ax.plot(sklearn_LDA_projection[II_0], np.zeros(len(II_0)), linestyle='None', marker='o', markersize=marker_size, color='blue', label='NSCLC')
ax.plot(sklearn_LDA_projection[II_1], np.zeros(len(II_1)), linestyle='None', marker='o', markersize=marker_size, color='red', label='SCLC')
ax.legend()

fig.show()

# -----------------------------------------------------
# 7. LDA, PCA, and clustering
# -----------------------------------------------------
# PCA on cell line data
sklearn_PCA = PCA(n_components=2)
PCA_scores = sklearn_PCA.fit(data_in).transform(X)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('PCA of cell line data')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.plot(PCA_scores[0:20, 0], PCA_scores[0:20, 1], linestyle='None', color='blue', marker='o')
ax.plot(PCA_scores[20:40, 0], PCA_scores[20:40, 1], linestyle='None', color='red', marker='o')
fig.show()

# kmeans clustering of cell line data
sklearn_KMeans = KMeans(n_clusters=2, random_state=0)
sklearn_KMeans.fit(X)

print "KMeans clustering results: labels"
print sklearn_KMeans.labels_
print "KMeans clustering results: cluster centers"
print sklearn_KMeans.cluster_centers_

# hierarchical clustering
sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='euclidean')
sklearn_agglomerative_clustering.fit(X)
print "agglomerative clustering results: complete linkage, euclidean distance"
print sklearn_agglomerative_clustering.labels_

sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='euclidean')
sklearn_agglomerative_clustering.fit(X)
print "agglomerative clustering results: average linkage, euclidean distance"
print sklearn_agglomerative_clustering.labels_

sklearn_agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='manhattan')
sklearn_agglomerative_clustering.fit(X)
print "agglomerative clustering results: complete linkage, manhattan distance"
print sklearn_agglomerative_clustering.labels_

# -----------------------------------------------------
# 8. use logistic regression for classification of the cell line data
# -----------------------------------------------------
num_of_samples = X.shape[0]
num_of_features = X.shape[1]

sklearn_logistic_classifier = LogisticRegression(random_state=0)

prediction_all = np.zeros(num_of_samples)
# leave-one-out cross validation
for i in range(num_of_samples):
    cur_X_test = X[i, :]
    cur_X_test = cur_X_test.reshape((1, num_of_features))

    cur_X_train = np.delete(X, obj=i, axis=0)

    cur_y_test = y[i]

    cur_y_train = np.delete(y, obj=i)

    sklearn_logistic_classifier.fit(cur_X_train, cur_y_train)

    cur_y_prediction = sklearn_logistic_classifier.predict(cur_X_test)

    prediction_all[i] = cur_y_prediction

print "leave-one-out CV: prediction"
print prediction_all