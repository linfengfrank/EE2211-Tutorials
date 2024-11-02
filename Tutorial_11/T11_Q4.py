# Question 4
import numpy as np

# Data points
x = np.array([1, 1])
y = np.array([0, 1])
z = np.array([0, 0])

data_points = np.array([x, y, z])

# Initial centers
centers = np.array([x, y])

def k_means (data_points, centers, n_clusters, max_iterations=100 , tol=1e-4):
    for _ in range (max_iterations):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data_points[:, np.newaxis]-centers, axis=2), axis =1)
        # Update centroids to be the mean of the data points assigned to them
        new_centers = np. zeros ((n_clusters, data_points.shape[1]))
        # End if centroids no longer change
        for i in range (n_clusters):
            new_centers[i] = data_points[labels==i].mean(axis=0)
        if np.linalg.norm(new_centers-centers) < tol :
            break
        centers = new_centers
    return centers , labels

centers, labels = k_means(data_points, centers, n_clusters=2)
print ("Converged centers :", centers )