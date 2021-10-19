# code to determine k for K-Means
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dataset_stanza import *


base_dir = 'datasets/rest/'
data = Dataset(base_dir, is_preprocessed=True)

distortions = []
K = range(3, 50)
for k in K:
    ac, vectors = aspect_cluster(data, k)
    dist = sum(np.min(cdist(vectors, ac.kmeans.cluster_centers_, 'cosine'), axis=1)) / vectors.shape[0]
    distortions.append(dist)

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('find the best k')
plt.savefig('./best_k.jpg')
plt.show()
