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
def do_PCA(data_frame):
    
    #data_frame = data.iloc[:,1:3]
    #Variance of each coloumn
    #data_frame.var()
    #calculating the mean vector
    #mean_vec = np.mean(data_frame)
    #print('mean vector\n',mean_vec)
    
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
    #splitting into a 2-dimentional matrix
    matrix_w = np.hstack((eig_pairs[0][1].reshape(2,1)))
    #Calculating new Eigen Spaces
    eig_space = matrix_w.dot(data_frame.T)
    print('Transformed matrix\n',eig_space)
    pca_results = {'data': data_frame,
                   'PC_variance': eig_val,
                   'eig_pairs':eig_pairs,
                   'scores': pca_scores,
                   'matrix_w': matrix_w,
                   'transformed_matrix':eig_space}
    #calculating the PC1 variance and PC2 variance
    PC1_variance = 100 * (pca_results['PC_variance'][1]) / sum(pca_results['PC_variance'])
    print('PC1_variance(%)=',PC1_variance)
    PC2_variance = 100 * (pca_results['PC_variance'][0]) / sum(pca_results['PC_variance'])
    print('PC2_variance(%)=',PC2_variance)
    print(pca_results)
    return pca_results

#LDA function
def my_LDA_two_class(data):
    
   #X = data.as_matrix()
   #y = np.concatenate((np.zeros(20), np.ones(20)))

   #II0 = np.where(y==0)
   #II1 = np.where(y==1)

   #II0 = II0[0]
   #II1 = II1[0]
   #x1 = X[II0, :]
   #x2 = X[II1, :]

   #X1 = x1.transpose()
   #X2 = x2.transpose()
   ##mean_x1=np.mean(X1,axis=1)
   #mean_x1=np.mean(X1,axis=1)
   #computing d dimensional mean vectors
  
   #computing the winthin class scatter matrix
   
    c1=[]
    c0=[]
    n=len(data)
    for i in range(n):
        if data.iloc[i,2]==0:
            c0.append(data.iloc[i,:2])
        else:
            c1.append(data.iloc[i,:2])

        
    c0_mean=np.mean(c0, axis=0)
    c1_mean=np.mean(c1, axis=0)
    c1_mean,c0_mean
    mean_vector=np.array([c0_mean,c1_mean])
    mean_vector[0]
    #mean_x1 = np.mean(c0,axis=1).reshape(2,1)
    #mean_x2 = np.mean(c1,axis=1).reshape(2,1)
   
    #s_x1 = np.dot((c0-mean_x1),(c0-mean_x1).T)
    #s_x2 = np.dot((c1-mean_x2),(c1-mean_x2).T)
    
    #within-class similarity
    cov=0
    for i in range(n):
        if data.iloc[i,2]==0:
            d=data.iloc[i,:2].reshape(2,1)
            c=c0_mean.reshape(2,1)
            cov+=(d-c).dot((d-c).T)
        else:
            d=data.iloc[i,:2].reshape(2,1)
            c1=c1_mean.reshape(2,1)
            cov+=(d-c1).dot((d-c1).T)
    s_within=cov


    #inner class similarity (s-between)
    data_mean1=np.mean(data.iloc[:,:2], axis=0)
    length_c0=len(data[(data.iloc[:,2]==0)])
    length_c1=len(data[data.iloc[:,2]==1])
    s_between=0

    data_mean2=data_mean1.reshape(2,1)
    mm1=c1_mean.reshape(2,1)
    mm0=c0_mean.reshape(2,1)
    for i in range(len(mean_vector)):
        if i==0:
            s_between+=length_c0*((mm0-data_mean2).dot((mm0-data_mean2).T))
        else:
            s_between+=length_c1*((mm1-data_mean2).dot((mm1-data_mean2).T))

    eig_values, eig_vectors = np.linalg.eig(np.linalg.inv(s_within).dot(s_between))
    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:,i]) for i in range(len(eig_values))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    for i in eig_pairs:
        print(i[0])
    

    W = np.hstack((eig_pairs[0][1].reshape(2,1)))
    print('W matrix=', W.real)
    
    X1=(data.iloc[:,:2])
    X_transformed=X1.dot(W.T)
    
    percentVarianceExplained_lda1 = 100 * (eig_pairs[0][0]) / (eig_pairs[0][0]+eig_pairs[1][0])
    print ('Variance of LDA1 onto w axis(%)',percentVarianceExplained_lda1  )

    percentVarianceExplained_lda2 = 100 * (eig_pairs[1][0]) / (eig_pairs[0][0]+eig_pairs[1][0])
    print ('Variance of LDA2 onto w axis(%): ' , percentVarianceExplained_lda2)

    lda_return={'matrix_w_lda':W,
                'x_tran': X_transformed,
                'eig_pairs':eig_pairs
               }
    return lda_return


#reading the data from the csv file
data1 = pd.read_csv("C:\Datasets\dataset_1 (1).csv")
V1 = np.array(data1['V1'])
V2 = np.array(data1['V2'])
label = np.array(data1['label'])
length = len(V1)
x_bar = np.mean(V1)
y_bar = np.mean(V2)

#omitting the label column
#my_cols = set(data1.columns)
#my_cols.remove('label')
#data2 = data1[my_cols]
data = data1.iloc[:,0:2]

#Q1-1
#plotting the raw data
plt.scatter(x=V1, y=V2)
plt.title("Plot between V1 and V2")
plt.xlabel("V1")
plt.ylabel("V2")
plt.show()
print("we can see clear separation between V1 and V2")
#we can see a clear separation between V1 and V2


# do PCA on the dataset
pca_results=do_PCA(data)

#projecting raw data on to PC1 axis
transformed_pca = pca_results['transformed_matrix']
plt.scatter(transformed_pca[:30],data1.iloc[0:30,2])
plt.scatter(transformed_pca[30:],data1.iloc[30:,2])
plt.title("Projection of data onto PC1 axis")
print("Again,we can see very clear separation")
plt.show()

#do lda on dataset
lda_results = my_LDA_two_class(data1)
x_tran_lda = lda_results['x_tran'].T

#plotting the data onto LD1 axis
plt.scatter(x_tran_lda[:30],data1.iloc[0:30,2])
plt.scatter(x_tran_lda[30:],data1.iloc[30:,2])
plt.title("Projection of Data on to LD1 axis")
plt.show()

#Adding the PC1 axis to the raw data plot
print("Adding the PC1 axis to the raw data plot")
X_pca = pca_results['eig_pairs']
X_pca_1 = X_pca[0][1][0]
X_pca_2 = X_pca[0][1][1]
plt.scatter(data.iloc[0:30,0],data.iloc[0:30,1])
plt.scatter(data.iloc[30:,0],data.iloc[30:,1])
plt.plot([0,-40*X_pca_1], [0,-40*X_pca_2],color='red',linewidth=4)
plt.show()

eig_vec_lda = lda_results['eig_pairs']
matrix_lda = lda_results['matrix_w_lda']

#plotting the LDA axis onto the previous graph
print("Adding the LDA axis to the above plot")
plt.scatter(data.iloc[0:30,0],data.iloc[0:30,1])
plt.scatter(data.iloc[30:,0],data.iloc[30:,1])
plt.plot([0,-40*X_pca_1], [0,-40*X_pca_2],color='red',linewidth=4)
plt.plot([0,20*eig_vec_lda[0][1][0]], [0,20*eig_vec_lda[0][1][1]],color='green',linewidth=4)
plt.show()