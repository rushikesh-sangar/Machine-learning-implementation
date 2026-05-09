# Data preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\1 May\Data.csv')

# Independent vatiable
x = dataset.iloc[:, :-1].values

# Dependent variable
y = dataset.iloc[:, 3].values.astype(object)

# sklearn to fill missing numerical values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Impute categorical values for independent variables
from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

# Impute categorical values for dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)


  

