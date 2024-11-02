# Question 6, Part i

##--- Import necessary libraries ---#
import random as rd
import numpy as np # linear algebra
from matplotlib import pyplot as plt

##-- Generate data ---#
## Set three centers, the model should predict similar results
center_1 = np.array([2,2])
center_2 = np.array([4,4])
center_3 = np.array([6,1])

## Generate random data and center it to the three centers
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3
data = np.concatenate((data_1, data_2, data_3), axis = 0)


##--- Initialize k-means ---##
# Initialize the number of clusters
k = 3

# Forgy Initialize the centroids
centers = data[np.random.choice(len(data), k, replace=False)]

##--- Define the k-means function ---#
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

##--- Run k-means ---##
centers, labels = k_means(data, centers, n_clusters =k)
print ("Converged centers :", centers )

###--- Show The Clustering Results ---#
plt.title('Clustering Results ')
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap ='viridis', alpha=0.5)
plt.scatter(centers[:, 0], centers [:, 1], marker ='*', s=200 , c='k')
plt.show()


###--- Show The Clustering Results ---#
# plt.figure(figsize=(8, 6)) 
# plt.title('Clustering Results')
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='k')
# plt.show()




