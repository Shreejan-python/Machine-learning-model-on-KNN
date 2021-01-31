import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs

x, y = make_blobs(n_samples=400, centers=5, cluster_std=0.60, random_state=0)

#plt.scatter(x[:, 0], x[:, 1], s=20)
#plt.show()

kmeans = KMeans(n_clusters=5)
kmeans.fit(x)
y_kmeans = kmeans.predict(x)

plt.scatter(x[:, 0], x[:,1], c=y_kmeans, s=120, cmap='summer')
centers_ = kmeans.cluster_centers_
plt.scatter(centers_[:, 0], centers_[:, 1], c='blue', s=100, alpha=0.9)
plt.show()