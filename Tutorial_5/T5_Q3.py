import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import inv

# Define Data: Number of students (X) and number of books sold (y)
X = np.array([36, 28, 35, 39, 30, 30, 31, 38, 36, 38, 29, 26])
X = X.reshape(-1, 1) # reshape as a vertical vector 

y = np.array([31, 29, 34, 35, 29, 30, 30, 38, 34, 33, 29, 26])
y = y.reshape(-1, 1) # reshape as a vertical vector

b   = np.ones( (len(X), 1) )
X_b = np.hstack((b, X)) # add bias to the X matrix

# (a) Scatter plot
plt.scatter(X, y, color='blue', label='Training samples')
plt.title('Scatter Plot of Books Sold vs. Students Registered')
plt.xlabel('Number of Students Registered')
plt.ylabel('Number of Books Sold')
plt.grid(True)
plt.legend()

# (b) Linear Regression: Calculate w
w = inv(X_b.T@X_b)@X_b.T@y
print("w is"); print(w)

# draw the estimated line
X_t = np.linspace(5,40, 50)
X_t = X_t.reshape(-1, 1)
b_t = np.ones((len(X_t),1)) # generate a bias column vector

X_b_t = np.hstack((b_t, X_t))
y_b_t = X_b_t@w

plt.plot(X_t, y_b_t, color='red', label='Linear Regression')
plt.legend()
plt.show()

# (c) Predict books sold when 30 students are registered
X_new_30 = 30
y_pred_30 = np.array([ [1, X_new_30]]) @ w
print(y_pred_30)

# (d) Predict books sold when 5 students are registered
X_new_5 = 5
y_pred_5 = np.array([ [1, X_new_5]]) @ w
print(y_pred_5)