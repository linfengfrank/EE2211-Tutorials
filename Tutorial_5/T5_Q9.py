import numpy as np
from numpy.linalg import inv
X = np.array([[1, 3, -1, 0], [1, 5, 1, 2], [1, 9, -1, 3], [1, -6, 7, 2], [1, 3, -2, 0]])
Y = np.array([[1, -1], [-1, 0], [1, 2], [0, 3], [1, -2]])
W = inv(X.T @ X) @ X.T @ Y
print(W)
newX=np.array([1, 8, 0, 2])
newY=newX@W
print(newY)