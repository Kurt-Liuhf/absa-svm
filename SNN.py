from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from dataset_stanza import *
from sklearn.metrics import silhouette_score
from hyperopt import hp, tpe, STATUS_OK, fmin


class ParametersTuner(object):

    def __init__(self, X):
        self.X = X
        self.best_ss = 0
        self.best_noise_ratio = 1
        self.best_params = {}
        self.best_num_class = 1000000
        self.min_loss = 100000

    def space4snn(self):
        params_space = {
            'n_neighbors': hp.choice('n_neighbors', [i for i in range(2, 15)]),
            'eps': hp.uniform('eps', 0.01, 0.99),
            'min_samples': hp.choice('min_samples', [i for i in range(2, 15)]),
        }

        return params_space

    def loss(self, labels, params):
        # 轮廓系数
        ss = silhouette_score(self.X, labels)
        # 噪声比
        noise_count = sum([1 if x == -1 else 0 for x in labels])
        noise_ratio = noise_count / len(labels)
        # 类别个数
        num_class = len(set(labels))

        loss = 0.1 * noise_ratio - 0.8 * ss + 0.1 * num_class * 0.1
        if loss < self.min_loss:
            self.best_ss = ss
            self.best_noise_ratio = noise_ratio
            self.min_loss = loss
            self.best_num_class = num_class
            self.best_params = params
            print('##########################################################')
            print('best_ss: %.5f' % self.best_ss)
            print('best_noise_ratio: %.5f' % self.best_noise_ratio)
            print('num_class: %d' % self.best_num_class)
            print('min_loss: %.5f' % self.min_loss)
            print('best params: %s' % str(params))
            print('##########################################################\n')

        return loss

    def snn(self, params):
        cluster = SharedNearestNeighbor(**params)
        labels = cluster.fit(self.X)

        return self.loss(labels, params)

    def tune_params(self, n_iter=5000):
        fmin(fn=self.snn,
            algo=tpe.suggest,
            space=self.space4snn(),
            max_evals=n_iter)


class SharedNearestNeighbor:

    def __init__(self, n_neighbors=7, eps=1, min_samples=5, weighted=False, n_jobs=5,
                 algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None):
        """
        :param n_neighbor:
        :param eps:
        :param min_samples:
        :param weighted:
        :param n_jobs:
        :param algorithm:
        :param leaf_size:
        :param metric:
        :param p:
        :param metric_params:
        """
        if weighted and eps >= 1:
            raise ValueError("For weighted SNN, please define a eps  value between 0 and 1.")
        if eps < 0:
            raise ValueError("Eps must be positive.")
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs, algorithm=algorithm,
                                      leaf_size=leaf_size, metric=metric, p=p, metric_params=metric_params)
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors
        self.labels_ = []
        self.components_ = []
        self.core_sample_indices = []
        self.weighted = weighted

    def _fit(self, X):
        if self.weighted:
            graph, max_similarity = self._inner_weighted(X)
        else:
            graph, max_similarity = self._inner(X)
        self.dbscan = DBSCAN(eps=max_similarity - self.eps, min_samples=self.min_samples,
                             metric='precomputed', n_jobs=self.n_jobs)
        self.dbscan.fit(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices = self.dbscan.core_sample_indices_

        return self.dbscan.labels_

    def fit(self, X):
        return self._fit(X)

    def _predict(self, X):
        return self.dbscan.fit_predict()

    def _inner_weighted(self, X):
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X, mode='distance')
        graph.data = np.reshape(self.n_neighbors - np.argsort(np.argsort(graph.data.reshape((-1, self.n_neighbors))))
                                , (-1, ))
        self.similarity_matrix = graph * graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0, 0]
        assert(max_similarity == np.max(self.similarity_matrix))
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data
        self.similarity_matrix.data = self.similarity_matrix.data / max(self.similarity_matrix.data)
        print('Biggest distance: ', np.max(self.similarity_matrix.data))
        print('Mean distance: ', np.mean(self.similarity_matrix.data))
        print('Median distance: ', np.median(self.similarity_matrix.data))
        max_similarity = 1

        return graph, max_similarity

    def _inner(self, X):
        self.neigh.fit(X)
        graph = self.neigh.kneighbors_graph(X)
        self.similarity_matrix = graph * graph.transpose()
        self.mask = self.similarity_matrix > self.eps
        max_similarity = self.similarity_matrix[0, 0]
        C = self.similarity_matrix.toarray()
        self.similarity_matrix.data = max_similarity - self.similarity_matrix.data

        A = graph.toarray()
        B = self.similarity_matrix.toarray()

        return graph, max_similarity

    def fit_predict(self, X):
        if self.weighted:
            graph, max_similarity = self._inner_weighted(X)
        else:
            graph, max_similarity = self._inner(X)

        self.dbscan = DBSCAN(eps=max_similarity - self.eps, min_samples=self.min_samples,
                             metric='precomputed', n_jobs=self.n_jobs)
        y = self.dbscan.fit_predict(self.similarity_matrix)
        self.labels_ = self.dbscan.labels_
        self.components_ = self.dbscan.components_
        self.core_sample_indices = self.dbscan.core_sample_indices_

        return y


if __name__ == "__main__":
    base_dir = 'datasets/rest/'
    data = Dataset(base_dir, is_preprocessed=True)
    k = 10
    ac, vectors = aspect_cluster(data, k)
    pt = ParametersTuner(vectors)
    pt.tune_params()
    # snn = SharedNearestNeighbor(n_neighbors=5, weighted=True, eps=0.5)
    # labels = snn.fit(vectors)
    # # 轮廓系数
    # ss = silhouette_score(vectors, labels)
    # # 噪声比
    # noise_count = sum([1 if x == -1 else 0 for x in labels])
    # noise_ratio = noise_count / len(labels)
    # print('轮廓系数: %.5f' % ss)
    # print('噪声比: %.5f', noise_ratio)
