import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
## get data from web
wine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-red.csv",sep=';')
wine.info()
y = wine.quality
x = wine.drop('quality',axis = 1)
## Include the offset/bias term
x0 = np.ones((len(y),1))
X = np.hstack((x0,x))
## split data into training and test sets
## (Note: this exercise introduces the basic protocol of using the training-test partitioning of samples for evaluation assuming the list of data is already randomly indexed)
## In case you really want a general random split to have a better training/test distributions:
## from sklearn.model_selection import train_test_split
## train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=99/1599, random_state = 0)
train_X = X[0:1500]
train_y = y[0:1500]
test_X = X[1500:1599]
test_y = y[1500:1599]
## linear regression
w = inv(train_X.T @ train_X) @ train_X.T @ train_y
print(w)
yt_est = test_X.dot(w);
MSE = np.square(np.subtract(test_y,yt_est)).mean()
print(MSE)
MSE = mean_squared_error(test_y,yt_est)
print(MSE)