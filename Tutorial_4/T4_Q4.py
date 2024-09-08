import numpy as np
from numpy.linalg import inv

X = np.array([[1, 0, 1, 0], [1, -1, 1, -1], [1,1,0,0]])
y = np.array([[1], [0], [1]])

w1 = X.T @ inv(X @ X.T) @y
print(w1)

# Another method
from numpy.linalg import pinv
w2 = pinv(X) @ y
print(w2)