import numpy as np
from sklearn.metrics import mean_squared_error

# Given data points
data = {
    1: 2, 0.8: 3, 2: 2.5, 2.5: 1, 3: 2.3, 4: 2.8, 4.2: 1.5, 
    6: 2.6, 6.3: 3.5, 7: 4, 8: 3.5, 8.2: 5, 9: 4.5
}

# Decision threshold
threshold = 5.0

# Split the data into two groups based on the decision threshold
group1 = {x: y for x, y in data.items() if x <= threshold}
group2 = {x: y for x, y in data.items() if x > threshold}

# Calculate the MSE for each group
mse_group1 = mean_squared_error(list(group1.values()), [np.mean(list(group1.values()))] * len(group1))
mse_group2 = mean_squared_error(list(group2.values()), [np.mean(list(group2.values()))] * len(group2))

# Compute the overall MSE by averaging the MSEs of the two groups
overall_mse = (mse_group1 + mse_group2) / 2

overall_mse_root = mean_squared_error(list(data.values()), [np.mean(list(data.values()))] * len(data))

print(f"Overall MSE at Root: {overall_mse_root}")
print(f"Overall MSE at Depth 1: {overall_mse}")