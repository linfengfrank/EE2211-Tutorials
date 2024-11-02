#Question 7
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np

# load the iris dataset
iris = load_iris ()

# get the data and the true labels
data = iris.data
y_true = iris.target

##--- Initialize k-means ---##
# Initialize the number of clusters
k = 3

# Forgy Initialize the centroids
centers = data[np.random.choice(len(data), k, replace=False)]

##--- Define the k-means function ---#
def k_means(data_points, centers, n_clusters, max_iterations=1000, tol=1e-6):
    for _ in range (max_iterations):
        # Assign each data point to the closest centroid
        labels = np.argmin(np.linalg.norm(data_points[:, np.newaxis]-centers, axis=2), axis =1)
        # Update centroids to be the mean of the data points assigned to them
        new_centers = np.zeros ((n_clusters,data_points.shape[1]) )
        # End if centroids no longer change
        for i in range (n_clusters):
            new_centers[i] = data_points[labels==i].mean(axis =0)
        if np.linalg.norm(new_centers-centers) < tol :
            break
        centers = new_centers
    return centers , labels

##--- Run k-means ---##
centers, y_pred = k_means(data, centers, n_clusters =k)

# create a mask that selects elements where the value is 0, 1, 2
mask_0 = ( y_pred == 0)
mask_1 = ( y_pred == 1)
mask_2 = ( y_pred == 2)
print('mask_0')
print(mask_0)

y_pred0=y_pred.copy()
y_pred0[mask_0] = 0
y_pred0[mask_1] = 1
y_pred0[mask_2] = 2

y_pred1=y_pred.copy ()
y_pred1[mask_0] = 0
y_pred1[mask_1] = 2
y_pred1[mask_2] = 1

y_pred2 = y_pred.copy ()
y_pred2[mask_0] = 1
y_pred2[mask_1] = 0
y_pred2[mask_2] = 2

y_pred3 = y_pred.copy ()
y_pred3[mask_0] = 1
y_pred3[mask_1] = 2
y_pred3[mask_2] = 0

y_pred4=y_pred.copy ()
y_pred4[mask_0 ] = 2
y_pred4[mask_1 ] = 0
y_pred4[mask_2 ] = 1

y_pred5 = y_pred.copy ()
y_pred5[mask_0] = 2
y_pred5[mask_1] = 1
y_pred5[mask_2] = 0

# calculate the accuracy of the clustering
accuracy = 0.0
for pred in [y_pred0, y_pred1, y_pred2, y_pred3, y_pred4, y_pred5 ]:
    accuracy = max ([ accuracy_score ( y_true , pred ), accuracy ])

print ("Accuracy of clustering : {:.2f}".format(accuracy))