import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Without bias, this is an even-determined system and X is invertible.
X   = np.array([[1, 0, 1], [2, -1, 1], [1, 1, 5]])
y = np.array([[1], [2], [3]])

b   = np.ones( (len(X),1) )
X_b = np.hstack((b, X)) # X matrix with bias

#(a) Perform a linear regression with addition of a bias/offset term
w = inv(X)@y
print(w)

X_t = np.array([[-1, 2, 8], [1, 5,-1]])
y_t = X_t@w
print(y_t)

#(b) After adding bias, it becomes an under-determined system.
w_b = X_b.T@inv(X_b@X_b.T)@y