import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score


digit = load_digits()
#print(digit.data.shape)

# print(type(digit))

k = KMeans(n_clusters=10, random_state=0)
clusters = k.fit_predict(digit.data)
#print(k.cluster_centers_.shape)

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centres = k.cluster_centers_.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centres):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    #print(axi.imshow)

label = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    label[mask] = mode(digit.target[mask])[0]


accuracy_score(digit.target, label)
print(accuracy_score)