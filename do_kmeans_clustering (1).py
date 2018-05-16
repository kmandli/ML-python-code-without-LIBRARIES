from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn import datasets

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------------------
# data file 1
# -----------------------------------------------------------------------------------------
np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names

# examine the data
# plot the data in 2D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(2, 1, 1)
II = (y==0)
ax.scatter(X[II,0], X[II, 1], color='blue')
II = (y==1)
ax.scatter(X[II,0], X[II, 1], color='red')
II = (y==2)
ax.scatter(X[II,0], X[II, 1], color='green')
ax.set_title('sepal')
ax.set_xlabel('length')
ax.set_ylabel('width')

ax = fig.add_subplot(2, 1, 2)
II = (y==0)
ax.scatter(X[II,2], X[II, 3], color='blue')
II = (y==1)
ax.scatter(X[II,2], X[II, 3], color='red')
II = (y==2)
ax.scatter(X[II,2], X[II, 3], color='green')
ax.set_title('petal')
ax.set_xlabel('length')
ax.set_ylabel('width')
fig.show()

# plot the data in 3D
fig = plt.figure(figsize=(10, 8))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
flower_name_and_label = [('Setosa', 0),
                         ('Versicolor', 1),
                         ('Virginiza', 2)]

for name, label in flower_name_and_label:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()


# kmeans clustering
estimators = [('k_means_iris_8', KMeans(n_clusters=8)),
              ('k_means_iris_3', KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1, init='random'))]

fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
#for name, est in estimators:
for i in range(len(estimators)):
    name = estimators[i][0]
    est = estimators[i][1]
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    
    fignum = fignum + 1

# -----------------------------------------------------------------------------------------
# data file 2
# -----------------------------------------------------------------------------------------
inFileName = "SCLC_study_output_filtered_2.csv"
dataIn = pd.read_csv(inFileName, header=0, index_col=0)

KMeans_result = KMeans(n_clusters=2, random_state=0).fit(dataIn)
KMeans_result.labels_
KMeans_result.cluster_centers_

k_means_result = k_means(dataIn, n_clusters=2, init='k-means++')