import numpy as np
m_list = [[1, 1], [3, 4]]
X = np.array(m_list)
inv_X = np.linalg.inv(X)
y = np.array([0, 1])
w = inv_X.dot(y)
print(w)


import matplotlib.pyplot as plt

# Define the range for w1
w1 = np.linspace(-3, 3, 400)

# Define the equations of the lines
w2_1 = -w1  # From w1 + w2 = 0, w2 = -w1
w2_2 = (1 - 3*w1) / 4  # From 3w1 + 4w2 = 1, w2 = (1 - 3*w1) / 4

# Plot the lines
plt.plot(w1, w2_1, 'r-', label='Equation (1)')
plt.plot(w1, w2_2, 'b--', label='Equation (2)')

# Add labels and title
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Plot of the lines')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()