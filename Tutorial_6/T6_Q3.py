# Tutorial 6, Question 3

import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[1,0,1], [1,-1,1]])
y = np.array([0, 1])

## Generate polynomial features
order = 3
poly = PolynomialFeatures(order)
P = poly.fit_transform(X)

## dual solution (without ridge)
w_dual = P.T @ inv(P @ P.T) @ y
print(w_dual)

# dual ridge regression
reg_L2 = 0.0001*np.identity(P.shape[0]) # lambda = 0.0001
w_dual_ridge = P.T @ (inv(P @ P.T + reg_L2)) @ y
print("Dual Ridge ")
print(w_dual_ridge)

## primal ridge
reg_L = 0.0001*np.identity(P.shape[1])
w_primal_ridge = inv(P.T @ P + reg_L) @ P.T @ y
print(w_primal_ridge)