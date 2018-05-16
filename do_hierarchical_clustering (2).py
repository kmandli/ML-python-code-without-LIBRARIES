#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 09:17:08 2017

@author: kmandli
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn import datasets

from sklearn.decomposition import PCA


np.random.seed(0)

iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names

HC_model = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='euclidean')
HC_model.fit(X)
HC_model.labels_

# do PCA first
PCA_result = PCA(n_components=4).fit_transform(X)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(PCA_result[0:49,0], PCA_result[0:49, 1], color='blue')
ax.scatter(PCA_result[50:99,0], PCA_result[50:99, 1], color='red')
ax.scatter(PCA_result[100:149,0], PCA_result[100:149, 1], color='green')

# then do hierarchical clustering
HC_model_2 = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='euclidean')
HC_model_2.fit(PCA_result[:, 0:1])
HC_model_2.labels_

