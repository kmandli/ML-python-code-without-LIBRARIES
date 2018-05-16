# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:40:16 2017

@author: kavya
"""

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from scipy import optimize as op
from sklearn.metrics import accuracy_score  

#Logistic Regression

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Regularized cost function
def regCostFunction(theta, X, y, _lambda = 0.1):
    m = len(y)
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0 
    reg = (_lambda/(2*m)) * np.sum(tmp**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

#Regularized gradient function
def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = _lambda*tmp /m

    return ((1 / m) * X.T.dot(h - y)) + reg

#Optimal theta 
def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)
    
    return result.x


Species = ['Iris-versicolor', 'Iris-virginica']
iris = datasets.load_iris()
x1 = iris.data[50:150,:]
X = np.array(x1[:,2:4])
#Normalizing the data using MinMax normalization
scaler = MinMaxScaler()
scaler.fit(X)
X = (scaler.transform(X))
y= np.array(iris.target[50:150]).reshape(100,1)
#converting 1's to 0's and 2's to 1's
y[y==1]=0
y[y==2]=1
dataset = np.concatenate((X, y), axis=1)
# leave one out train and test split cross validation
loo = LeaveOneOut()
loo.get_n_splits(X)
error_list_train = []
count = 0
n =1
for train_index, test_index in loo.split(X):
    
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(X_train, X_test, y_train, y_test)

    #optTheta = logisticRegression(X_train, y_train, np.zeros((n + 1,1)))
    #print optTheta
    #diabetes_y_pred = logistic.predict(X_test)
    
    all_theta = np.zeros((2, n + 1))

    #One vs all
    i = 0
    for flower in Species:
        optTheta = logisticRegression(X_train, y_train, np.zeros((n + 1,1)))
        all_theta[i] = optTheta
        i += 1
    #Predictions
    P = sigmoid(X_train.dot(all_theta.T)) #probability for each flower
    P = P[:,0]

    pred_list =[]
    for i in P:
        if i > 0.5:
            pred_list.append(1)
        else:
            pred_list.append(0)

    accuracy = accuracy_score(y_train,pred_list) * 100
    #print("Test Accuracy ", accuracy , '%')
    error_rate = round(100 - accuracy)
    error_list_train.append(error_rate)
    print("error rate :",error_rate)

print("Error list for 100 training datasets:",error_list_train)
print("Average error rate:",(sum(error_list_train))/(len(error_list_train)))


    
#print(error_list)