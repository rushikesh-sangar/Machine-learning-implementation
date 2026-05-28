# Simple linear regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\4 May\Salary data.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)

comparison = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
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
