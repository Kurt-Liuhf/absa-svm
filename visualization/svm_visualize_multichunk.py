import sys
sys.path.append('..')

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from dataset_depparse import *
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
data = Dataset(base_dir=base_dir, is_preprocessed=True)

def get_multichunk_data(aspect_id):
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=False)
    print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                                  (aspect_id, len(train_data), len(test_data)))
    x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, 'parse+chi')
    print(x_train.shape)
    scaler = Normalizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print(x_train.shape)

    # Take the first two features. We could avoid this by using a two-dim dataset
    X = x_train
    y = y_train
    tsne = TSNE(n_components=2, init='pca', n_iter=4000, random_state=42)
    decomposition_data = tsne.fit_transform(x_train)
    x_decompos = [x[0] for x in decomposition_data]
    y_decompos = [x[1] for x in decomposition_data]

    return decomposition_data, x_decompos, y_decompos, y

decom_data0, x_decompos0, y_decompos0, y0 = get_multichunk_data(0)
decom_data1, x_decompos1, y_decompos1, y1 = get_multichunk_data(1)
x_input = []
y_input = []
x0 = []
x1 = []
for i,j,k,l in zip(decom_data0, x_decompos0, y_decompos0, y0):
    x_input.append(i)
    y_input.append(l)
    x0.append(j)
    x1.append(k)
for i,j,k,l in zip(decom_data1, x_decompos1, y_decompos1, y1):
    x_input.append(i)
    y_input.append(l)
    x0.append(j)
    x1.append(k)


# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 10 # SVM regularization parameter
clf = svm.SVC(kernel='rbf', degree=3, gamma='auto', C=C, random_state=42)
clf.fit(x_input, y_input)
# models = (svm.SVC(kernel='linear', C=C, random_state=42),
#           svm.LinearSVC(C=C, max_iter=10000, random_state=42),
#           svm.SVC(kernel='rbf', gamma=0.7, C=C, random_state=42),
#           svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, random_state=42))
# models = (clf.fit(x_input, y_input) for clf in models)

# Set-up 2x2 grid for plotting.
fig, ax = plt.subplots()
plt.subplots_adjust(wspace=0.4, hspace=0.4)


X0, X1 = np.array(x0), np.array(x1)
xx, yy = make_meshgrid(X0, X1)

x0.extend([-7.0699, -7.31958, -6.47068, -693674])
x1.extend([-3.77695, -12.445, -4.68656, -11.0538])
y_input.extend([0, 0, 0, 0])
print(1)
plot_contours(ax, clf, xx, yy, cmap='Blues', alpha=0.8)
ax.scatter(x0, x1, c=y_input, cmap='Blues', s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
#
#
# plt.show()
plt.tight_layout()
pdf = PdfPages('svm_chunka+b_rbf.pdf')
pdf.savefig()
plt.close()
pdf.close()

