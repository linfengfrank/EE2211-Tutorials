import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

# define the matrix X, bias vector and y
X   = np.array([[-10], [-8], [-3], [-1], [2], [8]])
b   = np.ones( (len(X),1) )
X_b = np.hstack((b, X)) # X matrix with bias

y = np.array([[5], [5], [4], [3], [2], [2]])

#(a) Perform a linear regression with addition of a bias/offset term
w_b = inv(X_b.T@X_b)@X_b.T@y

#(b) Perform a linear regression without inclusion of any bias/offset term
w = inv(X.T@X)@X.T@y


# show the effect of adding a bias/offset term
X_test = np.linspace(-10,10, 400)
X_test = X_test.reshape(-1, 1)
b_test = np.ones((len(X_test),1)) # generate a bias column vector

X_b_test = np.hstack((b_test, X_test))
y_b_test = X_b_test@w_b 

plt.figure()
plt.plot(X_test, y_b_test, color='red', label='Linear Regression')
plt.plot(X, y, 'o', label='Training Samples')
plt.xlim(-10,10)
plt.ylim(-6, 6)
plt.grid(True)
plt.legend()
# plt.show()

# show the effect without a bias/offset term
y_test = X_test@w 

plt.figure()
plt.plot(X_test, y_test, color='red', label='Linear Regression')
plt.plot(X, y, 'o', label='Training Samples')

plt.xlim(-10,10)
plt.ylim(-6, 6)
plt.grid(True)
plt.legend()
plt.show()