# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 12:18:01 2017

@author: kavya
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
#from sklearn.decomposition import PCA

#Question 1 a and b
#creating LDA function with theory in class and applying the dataset on it
def my_LDA_two_class():


  
   dataset = pd.read_csv("C:\Datasets\SCLC_study_output_filtered_2.csv",index_col=0)

   #obtaining X and scaler y by projecting X on a line
   X = dataset.as_matrix()
   y = np.concatenate((np.zeros(20), np.ones(20)))

   II0 = np.where(y==0)
   II1 = np.where(y==1)

   II0 = II0[0]
   II1 = II1[0]
   x1 = X[II0, :]
   x2 = X[II1, :]

   X1 = x1.transpose()
   X2 = x2.transpose()
   #mean_x1=np.mean(X1,axis=1)
   #mean_x1=np.mean(X1,axis=1)
   #computing d dimensional mean vectors
   mean_x1 = np.mean(X1,axis=1).reshape(19,1)
   mean_x2 = np.mean(X2,axis=1).reshape(19,1)
   
   s_x1 = np.dot((X1-mean_x1),(X1-mean_x1).T)
   s_x2 = np.dot((X2-mean_x2),(X2-mean_x2).T)
   #computing the winthin class scatter matrix
   s_within = s_x1 + s_x2
   print(s_within)
   
   W=np.dot(LA.inv(s_within),(mean_x1-mean_x2))
   mu_1_tilde = np.dot(W.T,mean_x1)
   mu_2_tilde = np.dot(W.T,mean_x2)

   return mu_1_tilde,mu_2_tilde,X,W,X1,X2, x1, x2

mu_1_tilde,mu_2_tilde,X,W,X1, X2, x1, x2 = my_LDA_two_class()
W_scaled = W * 12.0 / W[0]
# projections
projection_1 = np.matmul(W.T, X1)
projection_2 = np.matmul(W.T, X2)
projection = np.matmul(X, W)

# slope of W
theta = math.atan(W[1] / W[0])


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(projection[20:40], np.zeros(20), linestyle='None', marker='o', markersize=10, color='blue', label='SCLC')
ax.plot(projection[0:20], np.zeros(20), linestyle='None', marker='o', markersize=10, color='black', label='NSCLC')
ax.plot(mu_1_tilde, 0.0, marker='X', markersize=15, color='blue')
ax.plot(mu_2_tilde, 0.0, marker='X', markersize=15, color='black')
ax.legend()
ax.set_title('Applying custom LDA and plotting results')
ax.set_xlabel('Projections')

fig.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_title('2D plot')
ax.scatter(x1[:, 0], x1[:, 1], color='black')
ax.scatter(x2[:, 0], x2[:, 1], color='blue')
ax.plot([0, W_scaled[0]], [0, W_scaled[1]], color='green')
ax.plot(-np.array([projection_1])*math.cos(theta), -np.array([projection_1])*math.sin(theta), color='blue', marker='x', markersize=15)
ax.plot(-np.array([projection_2])*math.cos(theta), -np.array([projection_2])*math.sin(theta), color='red', marker='x', markersize=15)
ax.set_aspect(1)

fig.show()




#comparing with sklearn.discriminantanalysis.LinearDiscriminantAnalysis.
#implementing LDA using sklearn.discriminantanalysis.LinearDiscriminantAnalysis.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


dataset = pd.read_csv("C:\Datasets\SCLC_study_output_filtered_2.csv",index_col=0)

   #obtaining X and scaler y by projecting X on a line
X = dataset.as_matrix()
y = np.concatenate((np.zeros(20), np.ones(20)))

II0 = np.where(y==0)
II1 = np.where(y==1)

II0 = II0[0]
II1 = II1[0]
x1 = X[II0, :]
x2 = X[II1, :]
   
#W, mu_1_tilde, mu_2_tilde = LinearDiscriminantAnalysis(x1, x2)
lda = LDA(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
#X_r2.coef_

#plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
#              label=target_name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
#plt.title('LDA')

#plt.show()

plt.scatter(x=np.real(X_r2 [range(40), 0]),y=np.zeros(40))
plt.title('LDA with sklearn')
plt.xlabel('LDA-1')
plt.ylabel('LDA-2')
plt.show()

   

