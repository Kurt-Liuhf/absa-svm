from sklearn.cluster import KMeans
import pickle
import numpy as np
from properties.properties import theproperty
class Cluster(object):

    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters,
                             max_iter=30000,
                             random_state=42)

    def _fit(self, X):
        """ X is a nd-array, shape=[#instance, dim]"""
        self.kmeans.fit(X)
        return self.kmeans.labels_

    def _predict(self, X):
        """ X is an nd-array, shape=[#instance, dim]"""
        return self.kmeans.predict(X)


class WordCluster(Cluster):
    def __init__(self, dataset, n_clusters=10):
        if len(dataset) < n_clusters * 2:
            n_clusters = n_clusters // 3
        super().__init__(n_clusters)
        self.dataset = [words.split(" ") for words in dataset]
        self.embed_dict = None    

    def avg_embed(self):
        embed_path = theproperty.rootpath + "text/embed.plk"#os.path.join(self.dataset.base_dir, 'parsed_data/embed.plk')
        if self.embed_dict is None:
            self.embed_dict = pickle.load(open(embed_path, 'rb'))

        vectors = []
        num_unk = 0
        for aspect in self.dataset:
            tmp_vecs = []
            for w in aspect:
                if w in self.embed_dict.keys():
                    tmp_vecs.append(self.embed_dict[w])
                else:
                    num_unk += 1
                    tmp_vecs.append(self.embed_dict['UNK'])
            vectors.append(np.mean(tmp_vecs, axis=0))
        return np.asarray(vectors)

    def fit(self):
        vectors = self.avg_embed()
        labels = super()._fit(vectors)
        return labels
    
def processCluster(column):
    wc = WordCluster(column)
    newColumn = wc.fit()
    return newColumn