import numpy as np
import matplotlib.pyplot as plt

# Q = np.array([[1, 0], [0, 1.5]])
Q = np.array([[1, 0.5], [0.5, 15]])

[x1,x2] = np.meshgrid(np.linspace(-10,10,1001),np.linspace(-10,10,1001))
x_vals = np.linspace(-10, 10, 1001)
y_vals = np.linspace(-10, 10, 1001)
X, Y = np.meshgrid(x_vals, y_vals)

Z = Q[0,0]*X**2 + Q[1,1]*Y**2 + 2*Q[0,1]*X*Y 

cp = plt.contour(X, Y, Z, np.linspace(0,200,10))
x_iter = np.array([[5], [3]])
notConverged = 1

lambdas, v = np.linalg.eig(Q)
eta = 2/(np.max(lambdas)+np.min(lambdas)) # optimal step size
iter = 0
x = np.zeros([2,1000])

while notConverged and iter < 1e3:
    plt.plot(x_iter[0],x_iter[1],'gx')
    x[:,iter] = x_iter.T
    x_iter = x_iter - eta*Q.dot(x_iter)
    if (np.linalg.norm(x_iter) < 1e-8):
        notConverged = 0
    
    iter = iter + 1

print('iter=')
print(iter)
plt.plot(x[0,0:iter-1],x[1,0:iter-1],'b--')
plt.show()