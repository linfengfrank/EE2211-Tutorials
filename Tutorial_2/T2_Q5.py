#(a) 
from pandas import read_csv 
dataset = read_csv('pima-indians-diabetes.csv', header=None) 
print(dataset.describe())

#(b) 
print((dataset[[1,2,3,4,5]] == 0).sum()) 

#(c) 
import numpy 

# mark zero values as missing or NaN 
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN) 

# print the first 20 rows of data 
print(dataset.head(20))
print(dataset.isnull().sum())