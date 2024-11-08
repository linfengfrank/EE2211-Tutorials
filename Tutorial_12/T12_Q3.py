# Question 3
import numpy as np
X = np.array([[1,1,3],[1,2,2.5]])
W = np.array([[-1,0,1],[0,-1,0],[1,0,1]])

Ftemp = X@W
F1 = np.zeros((2,3))
for i in range(0,2,1):
    for j in range(0,3,1):
        F1[i,j] = max(0.0,Ftemp[i,j])
        
Ftemp = F1@W
F2 = np.zeros((2,3))
for i in range(0,2,1):
    for j in range(0,3,1):
        F2[i,j] = max(0.0,Ftemp[i,j])

print(F1)
print(F2)