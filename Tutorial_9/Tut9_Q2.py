#!/usr/bin/env python3

import numpy as np

def mse(y):
    return np.mean(np.power(y-np.mean(y),2))


# Given data
data = np.array([
    [1, 2], [0.8, 3], [2, 2.5], [2.5, 1], [3, 2.3], [4, 2.8], 
    [4.2, 1.5], [6, 2.6], [6.3, 3.5], [7, 4], [8, 3.5], [8.2, 5], [9, 4.5]
])

# Split data into x and y
x = data[:, 0]
y = data[:, 1]

# Decision threshold
threshold = 5.0

# Split the data into two groups based on the decision threshold
ind_left  = np.where( x<=threshold )  # indies of x values less than or equal to the threshold
ind_right = np.where( x> threshold )  # indies of x values greater than the threshold

y_left  =  y[ind_left]
y_right = y[ind_right]

j_1 = len(y_left)
j_2 = len(y_right)
N   = len(y)

# Compute the MSE for difference levels
level_1_mse = mse(y)

level_2_mse = (j_1/N)*mse(y_left) + (j_2/N)*mse(y_right)

print(f"The level 1 MSE is: {round(level_1_mse, 4)}")
print(f"The level 2 MSe is: {round(level_2_mse, 4)}")