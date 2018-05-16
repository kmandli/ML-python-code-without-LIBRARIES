import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.cross_validation import train_test_split

import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

# ====================================================================
# plotting parameters
# ====================================================================
fig_width = 8
fig_height = 6

marker_size = 10

# ====================================================================
# program
# ====================================================================
# get data
in_file_name = "./data/banking.csv"
data_in = pd.read_csv(in_file_name, header=0)

print list(data_in.columns)
data_in.head()

# randomly select 25% of the data for illustration
samples_to_select = np.int_(data_in.shape[0] * np.random.rand(10000))
data_in = data_in.iloc[samples_to_select, :]

# barplot for the dependent variable
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(x='y', data=data_in, palette='hls')
fig.show()

# check the missing values
data_in.isnull().sum()

# customer job distribution
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(y='job', data=data_in)
fig.show()

# customer marital status distribution
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(x='marital', data=data_in)
fig.show()

# customer credit in default
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(x='default', data=data_in)
fig.show()

# customer housing loan
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(x='housing', data=data_in)
fig.show()

# customer personal loan
fig = plt.figure(figsize=(fig_width, fig_height))
sns.countplot(x='loan', data=data_in)
fig.show()

# use a small number of features for building the logistic regression model
data_for_analysis = data_in[['job', 'marital', 'default', 'housing', 'loan', 'poutcome', 'y']]

# Create dummy variables
data_for_analysis_dummy = pd.get_dummies(data_for_analysis,
                                         columns=['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data_for_analysis_dummy.head()

# remove columns with unknown values
all_columns = list(data_for_analysis_dummy)
indices_for_unknown_column = [i for i, s in enumerate(all_columns) if 'unknown' in s]

data_for_analysis_final = data_for_analysis_dummy.drop(data_for_analysis_dummy.columns[indices_for_unknown_column], axis=1)

# check the independence between independent variables
fig = plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(data_for_analysis_final.corr())
fig.show()

# get training and testing sets
X = data_for_analysis_final.iloc[:, 1:]
y = data_for_analysis_final.iloc[:, 0]

num_of_samples = X.shape[0]
num_of_test_samples = int(0.20 * num_of_samples)

test_sample_index = np.int_(num_of_samples * np.random.rand(num_of_test_samples))
train_sample_index = np.setdiff1d(np.arange(0, num_of_samples, 1), test_sample_index)
X_train = X.iloc[train_sample_index, :]
X_test = X.iloc[test_sample_index, :]
y_train = y.iloc[train_sample_index]
y_test = y.iloc[test_sample_index]

# ====================================================================
# logistic regression using sklearn
# ====================================================================
logistic_classifier = LogisticRegression(random_state=0)
logistic_classifier.fit(X_train, y_train)
print "intercept:"
print logistic_classifier.intercept_
print "coefficients:"
print logistic_classifier.coef_

# predict
y_prediction = logistic_classifier.predict(X_test)

# classifier performance
confusion_matrix_for_logistic_classifier = confusion_matrix(y_test, y_prediction)
print "confusion matrix:"
print confusion_matrix_for_logistic_classifier
print "classification report:"
print classification_report(y_test, y_prediction)

# ====================================================================
# logistic regression using gradient descent
# ====================================================================
m = X_train.shape[0]

# add the x_0 column bo both X_train and X_test
x_0 = pd.Series(np.ones(m), index=X_train.index)
X_train = X_train.assign(x_0=x_0.values)

all_columns = X_train.columns.tolist()
all_columns = all_columns[-1:] + all_columns[:-1]

X_train = X_train[all_columns]

x_0 = pd.Series(np.ones(num_of_test_samples), index=X_test.index)
X_test = X_test.assign(x_0=x_0.values)

all_columns = X_test.columns.tolist()
all_columns = all_columns[-1:] + all_columns[:-1]

X_test = X_test[all_columns]

n = X_train.shape[1]

# convert from pandas DataFrame to ndarray
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_test = y_test.as_matrix()

# gradient descent
iteration = 200

theta = np.zeros((iteration, n))
# theta[0, :] = 2.5 * np.ones(n)
initial_theta_0 = logistic_classifier.intercept_
initial_theta_1 = logistic_classifier.coef_[0]
initial_theta = 1.0 + np.concatenate((initial_theta_0, initial_theta_1))

theta[0, :] = initial_theta

J = np.zeros(iteration)

alpha = 5

for index_iter in range(iteration-1):
    if index_iter % 10 == 0:
        print "iteration " + str(index_iter)

    partial_derivative = np.zeros(n)

    for index_sample in range(m):
        cur_z = sum(X_train[index_sample, :] * theta[index_iter, :])
        cur_y_hat = 1.0 / (1.0 + np.exp(-cur_z))
        cur_residual = cur_y_hat - y_train[index_sample]

        partial_derivative = partial_derivative + X_train[index_sample, :] * cur_residual

        cur_cost = y_train[index_sample] * np.log10(cur_y_hat) + (1.0-y_train[index_sample]) * np.log10(1.0 - cur_y_hat)
        J[index_iter] = J[index_iter] + cur_cost

    J[index_iter] = -J[index_iter] / m

    theta[index_iter+1, :] = theta[index_iter, :] - alpha * partial_derivative / m

# calculate the last J(theta)
for index_sample in range(m):
    cur_z = sum(X_train[index_sample, :] * theta[iteration-1, :])
    cur_y_hat = 1.0 / (1.0 + np.exp(-cur_z))
    # cur_residual = cur_y_hat - y_train[index_sample]

    cur_cost = y_train[index_sample] * np.log10(cur_y_hat) + (1.0-y_train[index_sample]) * np.log10(1.0 - cur_y_hat)
    J[iteration-1] = J[iteration-1] + cur_cost

J[iteration-1] = -J[iteration-1] / m

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
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_0$')
ax.scatter(range(iteration), theta[:, 0], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_1$')
ax.scatter(range(iteration), theta[:, 1], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_2$')
ax.scatter(range(iteration), theta[:, 2], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_3$')
ax.scatter(range(iteration), theta[:, 3], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_4$')
ax.scatter(range(iteration), theta[:, 4], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_5$')
ax.scatter(range(iteration), theta[:, 5], color='blue', s=marker_size)
fig.show()

fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_6$')
ax.scatter(range(iteration), theta[:, 6], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_7$')
ax.scatter(range(iteration), theta[:, 7], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_8$')
ax.scatter(range(iteration), theta[:, 8], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_9$')
ax.scatter(range(iteration), theta[:, 9], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{10}$')
ax.scatter(range(iteration), theta[:, 10], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{11}$')
ax.scatter(range(iteration), theta[:, 11], color='blue', s=marker_size)
fig.show()

# figure 3
fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{12}$')
ax.scatter(range(iteration), theta[:, 12], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{13}$')
ax.scatter(range(iteration), theta[:, 13], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{14}$')
ax.scatter(range(iteration), theta[:, 14], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{15}$')
ax.scatter(range(iteration), theta[:, 15], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{16}$')
ax.scatter(range(iteration), theta[:, 16], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{17}$')
ax.scatter(range(iteration), theta[:, 17], color='blue', s=marker_size)
fig.show()

# figure 4
fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_subplot(2, 3, 1)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{18}$')
ax.scatter(range(iteration), theta[:, 18], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 2)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{19}$')
ax.scatter(range(iteration), theta[:, 19], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 3)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{20}$')
ax.scatter(range(iteration), theta[:, 20], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 4)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{21}$')
ax.scatter(range(iteration), theta[:, 21], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 5)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{22}$')
ax.scatter(range(iteration), theta[:, 22], color='blue', s=marker_size)

ax = fig.add_subplot(2, 3, 6)
ax.set_xlabel(r'iteration')
ax.set_ylabel(r'$\theta_{23}$')
ax.scatter(range(iteration), theta[:, 23], color='blue', s=marker_size)
fig.show()

# prediction
final_theta = theta[-1, :]
final_theta = final_theta.reshape((len(final_theta), 1))
z_predict = np.matmul(X_test, final_theta)
y_predict_probability = 1.0 / (1.0 + np.exp(-z_predict))

y_predict_binary = np.zeros(len(y_predict_probability))
for i in range(len(y_predict_probability)):
    if y_predict_probability[i] >= 0.5:
        y_predict_binary[i] = 1.0
    else:
        y_predict_binary[i] = 0.0

# get the confusion matrix
TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(y_test)):
    if y_test[i] == 1:
        if y_predict_binary[i] == 1:
            TP = TP + 1
        else:
            FN = FN + 1
    else:
        if y_predict_binary[i] == 1:
            FP = FP + 1
        else:
            TN = TN + 1

confusion_matrix_for_my_logistic_classifier = confusion_matrix(y_test, y_predict_binary)
print 'confusion matrix:'
print confusion_matrix_for_my_logistic_classifier

sklearn_theta_0 = logistic_classifier.intercept_
sklearn_theta_1 = logistic_classifier.coef_[0]
sklearn_theta = np.concatenate((sklearn_theta_0, sklearn_theta_1))

z_predict = np.matmul(X_test, sklearn_theta)
y_predict_probability_sklearn = 1.0 / (1.0 + np.exp(-z_predict))

y_predict_binary_sklearn = np.zeros(len(y_predict_probability_sklearn))
for i in range(len(y_predict_probability_sklearn)):
    if y_predict_probability_sklearn[i] >= 0.5:
        y_predict_binary_sklearn[i] = 1.0
    else:
        y_predict_binary_sklearn[i] = 0.0

print confusion_matrix(y_test, y_predict_binary_sklearn)