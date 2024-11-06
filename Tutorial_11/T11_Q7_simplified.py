from sklearn.datasets import load_iris
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris_dataset = load_iris()
X = np.array(iris_dataset['data'])
y = np.array(iris_dataset['target'])
K = 3  # number of clusters

# Plot initial data distribution
plt.scatter(X[:, 0], X[:, 1], s=7)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Initial Data Distribution')
# plt.show()

# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=K, n_init=1000, init='random', max_iter=100, random_state=0)
kout = kmeans.fit(X)
labels = kmeans.labels_
print('labels')
print(labels)

# Calibrate each centroid to a corresponding target class
predOutput = np.zeros((K, 3))  # Count of each true label in each cluster
for jj, cluster_label in enumerate(labels):
    predOutput[cluster_label, y[jj]] += 1
print('predOutput')
print(predOutput)

# Assign each cluster the most common target class
predOutputIdx = np.argmax(predOutput, axis=1)
print('predOutputIdx')
print(predOutputIdx)

# Generate predictions based on calibrated centroids
y_pred = np.array([predOutputIdx[label] for label in labels])

# Calculate and print accuracy
accuracy = accuracy_score(y, y_pred) * 100
print('The K-means method accuracy is:', accuracy, '%')

# Plot the clustering results
plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']
for i in range(K):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f'Cluster {i+1}', alpha=0.7)

plt.scatter(kout.cluster_centers_[:, 0], kout.cluster_centers_[:, 1], marker='x', c='black', s=200, label='Centroids')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('K-means Clustering on Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()