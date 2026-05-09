# SLR salary prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import variation
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
import pickle
import os

# Load the dataset
dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\4 May\Salary data.csv')

# Split the data into independent and dependent variables
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split the dataset into training and testing sets (80% - 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Training the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the test set
y_pred = regressor.predict(x_test)
print(y_pred)

# Comparison between test set and predicted set for dependent variable 
comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison)

# Visualize the training set
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Visualize the test set
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
print(f'Predicted salary for 12 years of experience: ${y_12[0]:,.2f}')

y_20 = model_coef * 20 + model_const
print(f'Predicted salary for 20 years of experience: ${y_20[0]:,.2f}')

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
print(f'Training score: {bias:.2f}')

# Variance score
variance = regressor.score(x_test, y_test)
print(f'Testing score (R^2): {variance:.2f}')

# Mean squared error
train_mse = mean_squared_error(y_train, regressor.predict(x_train))
print(f'Training MSE: {train_mse:.2f}')
test_mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {test_mse:.2f}')

# Save the trained model to disk
filename = 'Linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print('Model has been pickled and saved as Linear_regression_model.pkl')

# Gives current working directory
os.getcwd()