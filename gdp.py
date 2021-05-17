from sklearn.linear_model import LinearRegression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('GDP-per-capita-in-the-uk-since-1270.csv')
# print(dataset)


X = dataset.iloc[:, 2:3].values #get a copy of dataset exclude last column
y = dataset.iloc[:, -1].values #get array of dataset in column 1st
print(X)
# print(y)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# print(X_train)
# print(X_test)
"""
# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

degree=19
regressor=make_pipeline(PolynomialFeatures(degree),LinearRegression())
regressor.fit(X_train,y_train)

# regressor = PolynomialFeatures(2)
# regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
viz_train = plt
viz_train.yscale('log')
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('GDP-per-capita VS Year (Training set)')
viz_train.xlabel('Year')
viz_train.ylabel('GDP-per-capita')
viz_train.show()

# Visualizing the Test set results
viz_test = plt
viz_test.yscale('log')
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('GDP-per-capita VS Year (Test set)')
viz_test.xlabel('Year')
viz_test.ylabel('GDP-per-capita')
viz_test.show()


# Visualizing the Test set results
X_pred=[[2010],[2020],[2030],[2040],[2050],[2060],[2070],[2080],[2090],[2100],[2110],[2120],[2130],[2140],[2150],[2160],[2170],[2180],[2190],[2200]]
viz_pred = plt
viz_pred.yscale('log')
viz_pred.scatter(X_test, y_test, color='red')
viz_pred.plot(X_pred, regressor.predict(X_pred), color='blue')
viz_pred.title('GDP-per-capita VS Year (Predictiom)')
viz_pred.xlabel('Year')
viz_pred.ylabel('GDP-per-capita')
viz_pred.show()


