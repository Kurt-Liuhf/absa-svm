import os
os.chdir("..")

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from dataset import *
from search_feature_comb import *
from matplotlib.backends.backend_pdf import PdfPages



def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
# iris = datasets.load_iris()
base_dir = 'datasets/rest/'
data = Dataset(base_dir, is_preprocessed=True)

k = 10  # number of clusters
ac, vectors = aspect_cluster(data, k)

# use TSNE for visualization
tsne = TSNE(n_components=2, init='pca', n_iter=250, random_state=42)
decomposition_data = tsne.fit_transform(vectors)
x_decompos = [x[0] for x in decomposition_data]
y_decompos = [x[1] for x in decomposition_data]


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
models = (ac.kmeans, ac.kmeans,ac.kmeans,ac.kmeans)


# title for the plots
titles = ('KMeans', 'KMeans', 'KMeans', 'KMeans')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = np.array(x_decompos), np.array(y_decompos)
xx, yy = make_meshgrid(X0, X1)



for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=ac.kmeans.labels_, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

# plt.show()
plt.tight_layout()
pdf = PdfPages('svm_of_chunk10.pdf')
pdf.savefig()
plt.close()
pdf.close()

