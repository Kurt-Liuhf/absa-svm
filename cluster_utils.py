from sklearn.cluster._kmeans import KMeans
import numpy as np
# from dataset import *
from file_utils import *
import pickle
import os
from file_utils import *


class Cluster(object):

    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self._updateKMeans(n_clusters)

    def _fit(self, X):
        """ X is a nd-array, shape=[#instance, dim]"""
        self.kmeans.fit(X)
        return self.kmeans.labels_

    def _predict(self, X):
        """ X is an nd-array, shape=[#instance, dim]"""
        return self.kmeans.predict(X)
    
    def _updateKMeans(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters,
                             max_iter=30000,
                             random_state=42)


class VectorCluster(Cluster):
    
    def __init__(self, dataset, n_clusters=20):
        super().__init__(n_clusters)
        self.dataset = dataset
        self.train_vectors = []
        self.test_vectors = []

    def fit(self):
        train_vectors = [s.vector for s in self.dataset.train_data]
        self.train_vectors = np.asarray(train_vectors)
        labels = super()._fit(self.train_vectors) 
        for sample, cluster_id in zip(self.dataset.train_data, labels):
            sample.cluster = cluster_id   

    def predict(self):
        test_vectors = [s.vector for s in self.dataset.test_data]
        self.test_vectors = np.asarray(test_vectors)
        labels = super()._predict(self.test_vectors)
        for sample, cluster_id in zip(self.dataset.test_data, labels):
            sample.cluster = cluster_id

class AspectCluster(Cluster):

    def __init__(self, dataset, n_clusters=20):
        super().__init__(n_clusters)
        self.dataset = dataset
        self.training_clusters = {}
        self.test_clusters = {}
        self.embed_dict = None
        self.setaspect(-1, 0)

    def avg_embed(self, aspect_words):
        embed_path = os.path.join(self.dataset.base_dir, 'parsed_data/embed.plk')
        if self.embed_dict is None:
            self.embed_dict = pickle.load(open(embed_path, 'rb'))

        vectors = []
        num_unk = 0
        for aspect in aspect_words:
            tmp_vecs = []
            for w in aspect:
                if w in self.embed_dict.keys():
                    tmp_vecs.append(self.embed_dict[w])
                else:
                    num_unk += 1
                    tmp_vecs.append(self.embed_dict['UNK'])
            # tmp_vecs = [self.embed_dict[w] if w in self.embed_dict.keys()
            #             else self.embed_dict['UNK'] for w in aspect]
            vectors.append(np.mean(tmp_vecs, axis=0))
        print("number of aspect: ", str(len(aspect_words)))
        print("number of unk: ", str(num_unk))

        return np.asarray(vectors)

    def fit(self):
        train_data = self.dataset.train_data
        if self.splitaspect > 0:
            aspect_words = [s.text.split(' ') for s in train_data]
        else: aspect_words = [s.aspect.split(' ') for s in train_data]
        vectors = self.avg_embed(aspect_words)
        uniqueval = len(np.unique(vectors, axis = 0))

        if uniqueval < self.n_clusters:
            super()._updateKMeans(uniqueval)
        labels = super()._fit(vectors)

        for sample, cluster_id in zip(train_data, labels):
            
            sample.aspect_cluster = cluster_id
            if self.splitaspect > 0:
                if cluster_id == 0: sample.aspect_cluster = self.splitaspect
                else: sample.aspect_cluster += self.currentaspect - 1
            self.training_clusters.setdefault(cluster_id, [])
            self.training_clusters[cluster_id].append(sample.aspect)

        return labels, vectors

    def predict(self):
        test_data = self.dataset.test_data
        if self.splitaspect > 0:
            aspect_words = [s.text.split(' ') for s in test_data]
        else: aspect_words = [s.aspect.split(' ') for s in test_data]
        vectors = self.avg_embed(aspect_words)
        labels = super()._predict(vectors)

        for sample, cluster_id in zip(test_data, labels):
            sample.aspect_cluster = cluster_id 
            if self.splitaspect > 0:
                if cluster_id == 0: sample.aspect_cluster = self.splitaspect
                else: sample.aspect_cluster += self.currentaspect - 1
            self.test_clusters.setdefault(cluster_id, [])
            self.test_clusters[cluster_id].append(sample.aspect)

        return labels

    def save_cluster_result(self):
        base_dir = os.path.join(self.dataset.base_dir, 'aspect_cluster', str(self.dataset.aspect_cluster))
        # remove_dirs(base_dir)
        if os.path.isdir(os.path.join(base_dir, 'train')) is False: 
            make_dirs(os.path.join(base_dir, 'train'))
        for key, value in self.training_clusters.items():
            with open(os.path.join(base_dir, 'train', 'cluster_' + str(key)), 'w') as f:
                f.write("\n".join(value))
        if os.path.isdir(os.path.join(base_dir, 'test')) is False: 
            make_dirs(os.path.join(base_dir, 'test'))
        for key, value in self.test_clusters.items():
            with open(os.path.join(base_dir, 'test', 'cluster_' + str(key)), 'w') as f:
                f.write("\n".join(value))
    
    def setaspect(self, splitaspect=-1, currentaspect = 0):
        self.splitaspect = splitaspect
        self.currentaspect = currentaspect
        
    
class WordsCluster(Cluster):

    def __init__(self, dataset, n_clusters=20):
        super().__init__(n_clusters)
        self.dataset = dataset
        self.embed_dict = None
        self.train_vectors = []
        self.test_vectors = []
        self.aspect_words = {}   # feature words belong to each aspect
        self.cluster_models = {}  # cluster models for each aspect
        self.stopwords = stop_words()

        self.load_embed_dict()
        self.get_aspect_words()
        self.fit()

    def load_embed_dict(self):
        embed_path = os.path.join(self.dataset.base_dir, 'parsed_data/embed.plk')
        if self.embed_dict is None:
            self.embed_dict = pickle.load(open(embed_path, 'rb'))

    def get_aspect_words(self):
        for sample in self.dataset.train_data:
            self.aspect_words.setdefault(sample.aspect_cluster, [])
            new_words = [w for w in sample.words if w not in self.stopwords]
            self.aspect_words[sample.aspect_cluster].extend(new_words)

        # remove duplicated words
        for key in self.aspect_words.keys():
            self.aspect_words[key] = list(set(self.aspect_words[key]))

    def wordslist2vec(self, words):
        vectors = [self.word2vec(w) for w in words]
        return np.asarray(vectors)

    def word2vec(self, word):
        vec = self.embed_dict[word] if word in self.embed_dict.keys() else self.embed_dict['UNK']
        return vec

    def fit(self):
        for cluster_id, words in self.aspect_words.items():
            vectors = self.wordslist2vec(words)
            tmp_kmeans = KMeans(min(self.n_clusters, len(vectors)), random_state=42, max_iter=3000)
            tmp_kmeans.fit(vectors)
            self.cluster_models[cluster_id] = tmp_kmeans

    def generate_vector(self):
        # for training set
        train_vectors = []
        for sample in self.dataset.train_data:
            cluster_id = sample.aspect_cluster
            tmp_vec = [0 for _ in range(self.n_clusters)]
            tmp_words = [w for w in sample.words if w not in self.stopwords]
            if len(tmp_words) > 0:
                vecs = self.wordslist2vec(tmp_words)
                labels_ = self.cluster_models[cluster_id].predict(vecs)
                for x in labels_:
                    tmp_vec[x] += 1
            sample.sbow_vec = tmp_vec
            train_vectors.append(tmp_vec)

        # for test set
        test_vectors = []
        for sample in self.dataset.test_data:
            cluster_id = sample.aspect_cluster
            tmp_vec = [0 for _ in range(self.n_clusters)]
            tmp_words = [w for w in sample.words if w not in self.stopwords]
            if len(tmp_words) > 0:
                # vecs = self.wordslist2vec(sample.words)
                vecs = self.wordslist2vec(tmp_words)
                if self.cluster_models.get(cluster_id) is None:
                    continue
                labels_ = self.cluster_models[cluster_id].predict(vecs)
                for x in labels_:
                    tmp_vec[x] += 1
            sample.sbow_vec = tmp_vec
            test_vectors.append(tmp_vec)

        self.train_vectors = np.asarray(train_vectors)
        self.test_vectors = np.asarray(test_vectors)


if __name__ == '__main__':
    base_dir = 'datasets/rest/'
    data = Dataset(base_dir, is_preprocessed=True)
    wc = WordsCluster(data)
    wc.generate_vector()
