# Question 5
import numpy as np

# Data points
x1 = np.array([0, 0])
x2 = np.array([0, 1])
x3 = np.array([1, 1])
x4 = np.array([1, 0])
x5 = np.array([3, 0])
x6 = np.array([3, 1])
x7 = np.array([4, 0])
x8 = np.array([4, 1])

data_points = np.array ([x1, x2, x3, x4, x5, x6, x7, x8 ])

# Initial centers
c1_init = np. array ([0, 0])
c2_init = np. array ([3, 0])

centers = np. array ([c1_init, c2_init])

def k_means (data_points, centers, n_clusters, max_iterations=100 , tol=1e-4):
    for _ in range (max_iterations):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data_points[:, np.newaxis]-centers, axis=2), axis =1)
        # Update centroids to be the mean of the data points assigned to them
        new_centers = np. zeros ((n_clusters,data_points.shape[1]) )
        # End if centroids no longer change
        for i in range (n_clusters):
            new_centers[i] = data_points[labels==i].mean(axis =0)
        if np.linalg.norm(new_centers-centers) < tol :
            break
        centers = new_centers
    return centers , labels

centers, labels = k_means( data_points, centers, n_clusters=2)
print ("Converged centers:", centers )