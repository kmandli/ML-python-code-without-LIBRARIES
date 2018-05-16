import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def d_lm(x, y, confidence=0.95):
    n = len(x)

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    S_yx = np.sum((y - y_bar) * (x - x_bar))
    S_xx = np.sum((x - x_bar)**2)

    # ====== estimate beta_0 and beta_1 ======
    beta_1_hat = S_yx / S_xx # also equal to (np.cov(x, y))[0, 1] / np.var(x)
    beta_0_hat = y_bar - beta_1_hat * x_bar

    # ====== estimate sigma ======
    # residual
    y_hat = beta_0_hat + beta_1_hat * x
    r = y - y_hat
    sigma_hat = np.sqrt(sum(r**2) / (n-2))

    # ====== estimate sum of squares ======
    # total sum of squares
    SS_total = np.sum((y - y_bar)**2)
    # regression sum of squares
    SS_reg = np.sum((y_hat - y_bar)**2)
    # residual sum of squares
    SS_err = np.sum((y - y_hat)**2)

    # ====== estimate R2: coefficient of determination ======
    R2 = SS_reg / SS_total

    # ====== estimate MS ======
    # sample variance
    MS_total = SS_total / (n-1)
    MS_reg = SS_reg / 1.0
    MS_err = SS_err / (n-2)

    # ====== estimate F statistic ======
    F = MS_reg / MS_err
    F_test_p_value = 1 - stats.f._cdf(F, dfn=1, dfd=n-2)

    # ====== beta_1_hat statistic ======
    beta_1_hat_var = sigma_hat**2 / ((n-1) * np.var(x))
    beta_1_hat_sd = np.sqrt(beta_1_hat_var)
    # confidence interval
    z = stats.t.ppf(q=0.025, df=n-2)
    beta_1_hat_CI_lower_bound = beta_1_hat - z * beta_1_hat_sd
    beta_1_hat_CI_upper_bound = beta_1_hat + z * beta_1_hat_sd
    # hypothesis tests for beta_1_hat
    # H0: beta_1 = 0
    # H1: beta_1 != 0
    beta_1_hat_t_statistic = beta_1_hat / beta_1_hat_sd
    beta_1_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_1_hat_t_statistic), df=n-2))

    # ====== beta_0_hat statistic ======
    beta_0_hat_var = beta_1_hat_var * np.sum(x**2) / n
    beta_0_hat_sd = np.sqrt(beta_0_hat_var)
    # confidence interval
    beta_0_hat_CI_lower_bound = beta_0_hat - z * beta_0_hat_sd
    beta_1_hat_CI_upper_bound = beta_0_hat + z * beta_0_hat_sd
    beta_0_hat_t_statistic = beta_0_hat / beta_0_hat_sd
    beta_0_hat_t_test_p_value = 2 * (1 - stats.t.cdf(np.abs(beta_0_hat_t_statistic), df=n-2))

    # confidence interval for the regression line
    sigma_i = 1.0/n * (1 + ((x - x_bar) / np.std(x))**2)
    y_hat_sd = sigma_hat * sigma_i

    y_hat_CI_lower_bound = y_hat - z * y_hat_sd
    y_hat_CI_upper_bound = y_hat + z * y_hat_sd

    lm_result = {}
    lm_result['beta_1_hat'] = beta_1_hat
    lm_result['beta_0_hat'] = beta_0_hat
    lm_result['sigma_hat'] = sigma_hat
    lm_result['y_hat'] = y_hat
    lm_result['R2'] = R2
    lm_result['F_statistic'] = F
    lm_result['F_test_p_value'] = F_test_p_value
    lm_result['MS_error'] = MS_err
    lm_result['beta_1_hat_CI'] = np.array([beta_1_hat_CI_lower_bound, beta_1_hat_CI_upper_bound])
    lm_result['beta_1_hat_standard_error'] = beta_1_hat_sd
    lm_result['beta_1_hat_t_statistic'] = beta_1_hat_t_statistic
    lm_result['beta_1_hat_t_test_p_value'] = beta_1_hat_t_test_p_value
    lm_result['beta_0_hat_standard_error'] = beta_0_hat_sd
    lm_result['beta_0_hat_t_statistic'] = beta_0_hat_t_statistic
    lm_result['beta_0_hat_t_test_p_value'] = beta_0_hat_t_test_p_value
    lm_result['y_hat_CI_lower_bound'] = y_hat_CI_lower_bound
    lm_result['y_hat_CI_upper_bound'] = y_hat_CI_upper_bound

    return lm_result

# --------------------------------------------------------------------------
# produce x, noise-free y, noisy-corrupted y
# --------------------------------------------------------------------------
n = 20
np.random.seed(0)

x = -2 + 4 * np.random.rand(n)
x = np.sort(x)

beta_0 = -4.0
beta_1 = 1.4
sigma = 0.5

epsilon = sigma * np.random.normal(loc=0.0, scale=1, size=n)

y_theoretical = beta_0 + beta_1 * x
y = beta_0 + beta_1 * x + epsilon

# dataIn = pd.read_csv('linear_regression_test_data.csv')
# x = np.array(dataIn['x'])
# y = np.array(dataIn['y'])
# y_theoretical = np.array(dataIn['y_theoretical'])

n = len(x)

x_bar = np.mean(x)
y_bar = np.mean(y)

# do linear regression using my own function
lm_d_result = d_lm(x, y)

# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x, y, color='red')
ax.plot(x, y_theoretical, color='green', label='theoretical')
ax.plot(x, lm_d_result['y_hat'], color='blue', label='predicted')
ax.plot(x, np.ones(n)*y_bar, color='black', linestyle=':')
ax.plot([x_bar, x_bar], [np.min(y), np.max(y)], color='black', linestyle=':')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='lower right', fontsize=9)
fig.show()
# upper right, upper left, lower right, lower left, center left, center right, upper center, lower center

# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
# 1. are r and y_hat uncorrelated?
r = y - lm_d_result['y_hat']
np.corrcoef(r, lm_d_result['y_hat'])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(lm_d_result['y_hat'], r, color='blue')
ax.set_xlabel('y_hat')
ax.set_ylabel('r')
fig.show

# do linear regression using sklearn
lm_sklearn= linear_model.LinearRegression()
x = x.reshape((len(x), 1))
lm_sklearn.fit(x, y)
y_hat = lm_sklearn.predict(x)

lm_sklearn_result = {}
lm_sklearn_result['beta_0_hat'] = lm_sklearn.intercept_
lm_sklearn_result['beta_1_hat'] = lm_sklearn.coef_
lm_sklearn_result['R2'] = r2_score(y, y_hat)
lm_sklearn_result['mean_squared_error'] = mean_squared_error(y, y_hat)
lm_sklearn_result['y_hat'] = y_hat
