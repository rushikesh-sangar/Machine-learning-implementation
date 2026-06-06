# Multiple linear regression

import numpy as np
import pandas as pd

dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\11 May\MLR\MLR investment\Investment.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]

X = pd.get_dummies(X, dtype = int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

X = np.append(arr = np.full((50, 1), 42467).astype(int), values = X, axis = 1)

import statsmodels.api as sm
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# Ordinary least squares
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

bias = regressor.score(X_train, y_train)
print(bias)

variance = regressor.score(X_test, y_test)
print(variance)