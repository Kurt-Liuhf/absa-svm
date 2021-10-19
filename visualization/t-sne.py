""" visualization for aspect clustering """
import sys
sys.path.append('..')

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from dataset_stanza import *

base_dir = 'datasets/rest/'
data = Dataset(base_dir, is_preprocessed=True)

k = 5  # number of clusters
ac, vectors = aspect_cluster(data, k)

# use TSNE for visualization
tsne = TSNE(n_components=2, init='pca', n_iter=4000, random_state=42)
decomposition_data = tsne.fit_transform(vectors)
x_decompos = [x[0] for x in decomposition_data]
y_decompos = [x[1] for x in decomposition_data]

# fig = plt.figure(figsize=(10, 8))
# ax = plt.axes()
# plt.scatter(x_decompos, y_decompos, c=ac.kmeans.labels_, marker='o')
# plt.xticks()
# plt.yticks()
# plt.show()
colors = ['aliceblue',
'antiquewhite',
'aqua',
'aquamarine',
'azure',
'beige',
'bisque',
'black',
'blanchedalmond',
'blue',
'blueviolet',
'brown',
'burlywood',
'cadetblue',
'chartreuse',
'chocolate',
'coral',
'cornflowerblue',
'cornsilk',
'crimson',
'cyan',
'darkblue',
'darkcyan',
'darkgoldenrod',
'darkgray',
'darkgreen',
'darkkhaki',
'goldenrod',
'gray',
'green',
'greenyellow']

markers = ['.',
',',
'o',
'v',
'^',
'<',
'>',
'1',
'2',
'3',
'4',
's',
'p',
'*',
'h',
'H',
'+',
'x',
'D',
'd',
'|',
'_']

cluster_labels = set(ac.kmeans.labels_)
mycolors = [colors[x] for x in ac.kmeans.labels_]
fig = plt.figure(figsize=(10, 8))
ax = plt.axes()
plt.scatter(x_decompos, y_decompos, c=ac.kmeans.labels_, marker='o')
plt.xticks()
plt.yticks()
plt.show()

# cluster_labels = set(ac.kmeans.labels_)
# print(cluster_labels)
# for cl in cluster_labels:
#     decom_x = []
#     decom_y = []
#     for decompos, label in zip(decomposition_data, ac.kmeans.labels_):
#         if label == cl:
#             decom_x.append(decompos[0])
#             decom_y.append(decompos[1])
#     plt.plot(decom_x, decom_y, color=colors[cl], marker=markers[cl])


