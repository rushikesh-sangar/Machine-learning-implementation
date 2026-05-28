# Feature scaling using normalization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\Full Stack Data Science With Gen AI & Agentic AI Notes\Notes\May\1 May\Data.csv')

# Independent vatiable
X = dataset.iloc[:, :-1].values

# Dependent variable
y = dataset.iloc[:, 3].values.astype(object)

# sklearn to fill missing numerical values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Impute categorical values for independent variables
from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Impute categorical values for dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Feature scaling using normalization

from sklearn.preprocessing import Normalizer
nz_X = Normalizer()
X_train = nz_X.fit_transform(X_train)
X_test = nz_X.transform(X_test)