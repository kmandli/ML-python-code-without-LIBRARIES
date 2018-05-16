# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:52:37 2017

@author: kavya
"""


import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import datasets
from sklearn import linear_model
get_ipython().magic('matplotlib inline')
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#importing the diabetes dataset
db = datasets.load_diabetes()

#selecting x and depending variable y
x_db = db.data[:, np.newaxis, 2]
y_db = db.target.reshape(442,1)
diab = np.concatenate((x_db, y_db), axis=1)

#splitting x and y into test and tarin datasets
#20 random points for test, rest for train data
train,test = train_test_split(diab, test_size=0.045)
X_Test,Y_Test=test.T
X_Train,Y_Train=train.T
X_Test = X_Test.reshape(20,1)
Y_Test = Y_Test.reshape(20,1)
X_Train = X_Train.reshape(422,1)
Y_Train = Y_Train.reshape(422,1)

#creating regression model and predicting using testing set
linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_Train, Y_Train)
Y_Prediction = linear_regression.predict(X_Test)

print('Coefficients: \n', linear_regression.coef_)
print("Mean squared error: %.2f" % mean_squared_error(Y_Test, Y_Prediction))
print('Variance score: %.2f' % r2_score(Y_Test, Y_Prediction))

#plotting testing x vs testing y and testing x vs predicted Y in same plot
plt.scatter(X_Test, Y_Test,  color='black')
plt.plot(X_Test, Y_Prediction, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
