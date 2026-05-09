# Simple linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\4 May\Salary data.csv')

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)
print(y_pred)

comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

model_coef = regressor.coef_
print(model_coef)

model_const = regressor.intercept_
print(model_const)

y_12 = model_coef * 12 + model_const
print(y_12)

y_20 = model_coef * 20 + model_const
print(y_20)

# Mean
dataset.mean()
dataset['Salary'].mean()

# Median
dataset.median()
dataset['Salary'].median()

# Mode
dataset['YearsExperience'].mode()
dataset['Salary'].mode()

# Variance
dataset.var()
dataset['Salary'].var()

# Standard deviation
dataset.std()
dataset['Salary'].std()

# Coefficient of variation (CV)
from scipy.stats import variation

variation(dataset.values)
variation(dataset['Salary'])

# Correlation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

# Skewness
dataset.skew()
dataset['Salary'].skew()

# Standard error
dataset.sem()
dataset['Salary'].sem()

# Z-score
import scipy.stats as stats

dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

# Degree of freedom
a = dataset.shape[0] # It gives number of rows
b = dataset.shape[1] # It gives number of columns
degree_of_freedom = a - b
print(degree_of_freedom)

# SSR (Sum of squares regression)
y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean) ** 2)
print(SSR)

# SSE (Sum of squares error)
y = y[0:6]
SSE = np.sum((y - y_pred) ** 2)
print(SSE)

# SST (Sum of squares total)
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values - mean_total) ** 2)
print(SST)

# R-Squared
r_squared = 1 - (SSR/SST)
print(r_squared)

# Bias score
bias = regressor.score(x_train, y_train)
print(bias)

# Variance score
variance = regressor.score(x_test, y_test)
print(variance)



