# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:09:10 2017

@author: kavya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
#PCA function

def do_PCA(data):
    
    data_frame = data.iloc[:,1:3]
    #Variance of each coloumn
    data_frame.var()
    #calculating the mean vector
    mean_vec = np.mean(data_frame)
    print('mean vector\n',mean_vec)
    
    
    #calculating varinace between X and Y
    cov_mat = np.cov(data_frame.T)
    print('Covariance Matrix  between X & Y\n',cov_mat)
    #calculating the Eigen values and Eigen vectors 
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    print('eigen values and eigen vectors\n',eig_val,eig_vec)
    #calculating key value pairs of the eigen vectors and eigen values
    eig_pairs = [(np.abs(eig_val[k]), eig_vec[:,k]) for k in range(len(eig_val))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    for k in eig_pairs:
        print(k[0])
    pca_scores = np.matmul(data_frame, eig_vec)
    print('PCA Scores\n',pca_scores)
    #defining the pca results for plotting
    pca_results = {'data': data_frame,
                   'PC_variance': eig_val,
                   'loadings':eig_vec,
                   'scores': pca_scores}
                   
    print(pca_results)
    #percentVarianceExplained = 100 * (pca_results['PC_variance'][1]) / sum(pca_results['PC_variance'])
    #splitting into a 2-dimentional matrix
    matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1)))
    #Calculating new Eigen Spaces
    eig_space = matrix_w.dot(data_frame.T)
    print('Transformed matrix\n',eig_space)

    return pca_results
"""
def do_PCA(x):
    columnMean = x.mean(axis=0)
    columnMeanAll = np.tile(columnMean, reps=(x.shape[0], 1))
    xMeanCentered = x - columnMeanAll

    # use mean_centered data or standardized mean_centered data
    dataForPca = xMeanCentered

    # get covariance matrix of the data
    covarianceMatrix = np.cov(dataForPca, rowvar=False)

    # eigendecomposition of the covariance matrix
    eigenValues, eigenVectors = LA.eig(covarianceMatrix)
    II = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[II]
    eigenVectors = eigenVectors[:, II]

    # get scores
    pcaScores = np.matmul(dataForPca, eigenVectors)

    # collect PCA results
    pcaResults = {'data': x,
                   'mean_centered_data': xMeanCentered,
                   'PC_variance': eigenValues,
                   'loadings': eigenVectors,
                   'scores': pcaScores}

    return pcaResults
"""

#linear regression function
def do_lr(x, y, confidence=0.95):
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)
    S_yx = np.sum((y - y_bar) * (x - x_bar))
    S_xx = np.sum((x - x_bar)**2)
    # calculating beta 0 and beta 1
    beta_1_hat = S_yx / S_xx # also equal to (np.cov(x, y))[0, 1] / np.var(x)
    beta_0_hat = y_bar - beta_1_hat * x_bar
    #calculating sigma hat
    y_hat = beta_0_hat + beta_1_hat * x
    r = y - y_hat
    sigma_hat = np.sqrt(sum(r**2) / (n-2))
    #calculating total sum of squares
    SS_total = np.sum((y - y_bar)**2)
    # regression sum of squares
    SS_reg = np.sum((y_hat - y_bar)**2)

    #estimate R2: coefficient of determination 
    R2 = SS_reg / SS_total
    
    lin_reg = {}
    lin_reg['beta_1_hat'] = beta_1_hat
    lin_reg['beta_0_hat'] = beta_0_hat
    lin_reg['sigma_hat'] = sigma_hat
    lin_reg['y_hat'] = y_hat
    lin_reg['R2'] = R2
             
    return lin_reg
    
#reading the data from the csv file
data1 = pd.read_csv("C:\Datasets\linear_regression_test_data.csv")
x = np.array(data1['x'])
y = np.array(data1['y'])
y_theoretical = np.array(data1['y_theoretical'])
length = len(x)
x_bar = np.mean(x)
y_bar = np.mean(y)

# do PCA and then linear regression using functions do_PCA and do_lr
pca_results=do_PCA(data1)
lm_d_result = do_lr(x, y)

"""
#plotting the graphs
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#x vs y
ax.scatter(x, y, color='red')
#x vs y-theoritical
ax.plot(x, y_theoretical, color='yellow')
#PC axis 1
plt.plot([0, 8*pca_results['loadings'][1, 1]], [0, 8*pca_results['loadings'][0, 1]],
           color='green',linewidth=3)
fig.show()
plt.title('question1-graph1')

#plot between the y_precited and the x values
x_plot = plt.scatter(data1.iloc[:,1], data1.iloc[:,2], c='b') 
#plot between the y_actual and the x values
y_plot=plt.scatter(data1.iloc[:,1], data1.iloc[:,3], c='r')
#PC axis 1
plt.plot([0, 8*pca_results['loadings'][1, 1]], [0, 8*pca_results['loadings'][0, 1]],
            color='green',linewidth=3)
#Regression line
plt.plot(x, lm_d_result['y_hat'], color='black')
plt.title('question1-graph2')
"""
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#x vs y
ax.scatter(x, y, color='red')
ax.plot(x, y_theoretical, color='yellow')
ax.plot(x, lm_d_result['y_hat'], color='black')
plt.plot([0, 5*pca_results['loadings'][0, 1]], [0, 5*pca_results['loadings'][1, 1]],
            color='green',linewidth=3)
fig.show()
plt.title('question 1-graph1')


x_plot = plt.scatter(data1.iloc[:,1], data1.iloc[:,2], c='blue') 
y_plot=plt.scatter(data1.iloc[:,1], data1.iloc[:,3], c='red') 
plt.plot([0, 5*pca_results['loadings'][0, 1]], [0, 5*pca_results['loadings'][1, 1]],
            color='green',linewidth=3)
plt.plot(x, lm_d_result['y_hat'], color='black')
plt.title('question 1-graph2')

