# Question 4
import numpy as np
beta = 1

X = np.array([[1,2,1],[1,5,1]])
W1 = np.array([[-1,0,1],[0,-1,0],[1,0,-1]])
W23 = np.array([[-1,0,1],[0,-1,0],[1,0,1],[1,-1,1]])

# sigmoid 1/1+exp(-beta*a)
Ftemp = X@W1
F1 = np.hstack((np.ones((2,1)),1/(1 + np.exp(-beta*Ftemp))))
print(F1)
Ftemp = F1@W23
F2 = np.hstack((np.ones((2,1)),1/(1 + np.exp(-beta*Ftemp))))
print(F2)
Ftemp = F2@W23
F3 = 1/(1 + np.exp(-beta*Ftemp))
print(F3)