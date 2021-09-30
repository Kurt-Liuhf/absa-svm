from sklearn.cluster import KMeans
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.svm import SVC
from sklearn.ensemble import *
from sklearn.preprocessing import Normalizer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from thundersvm import SVC as TSVC
import time


class Dataset:

    def __init__(self, n_clusters=5):
        self.train_set = []
        self.test_set = []
        self.kmeans = KMeans(n_clusters=n_clusters,
                             n_jobs=20,
                             max_iter=5000,
                             random_state=42)

    def load_data_from_file(self, path):
        pass

    def process_data(self, X_train, y_train, X_test, y_test):
        train_cluster_ids = self.split_via_cluster(X_train)
        test_cluster_ids = self.cluster_predict(X_test)
        for x, y, c in zip(X_train, y_train, train_cluster_ids):
            self.train_set.append(Sample(x, y, c))
        for x, y, c in zip(X_test, y_test, test_cluster_ids):
            self.test_set.append(Sample(x, y, c))

    def process_after_split(self, X_data, y_data, test_ratio=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                            test_size=test_ratio,
                                                            random_state=0)
        self.process_data(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    def split_via_cluster(self, X_data):
        self.kmeans.fit(X_data)
        return self.kmeans.labels_

    def cluster_predict(self, x):
        return self.kmeans.predict(x)

    def data_from_aspect(self, cluster_id, is_sampling=True):
        X_train = []
        y_train = []
        for sample in self.train_set:
            if sample.cluster_id == cluster_id:
                X_train.append(sample.x)
                y_train.append(sample.y)
        print(len(X_train))
        if is_sampling:
            label_counter = Counter(y_train)
            print(label_counter)
            mc_label = label_counter.most_common()[0][1]
            for label in label_counter.keys():
                label_count = label_counter[label]
                if label_count < mc_label:
                    for s in self.train_set:
                        if s.y == label and s.cluster_id != cluster_id:
                            X_train.append(s.x)
                            y_train.append(s.y)
                            label_count += 1
                        if label_count >= mc_label:
                            break
        print(len(X_train))
        X_test = [s.x for s in self.test_set if s.cluster_id == cluster_id]
        y_test = [s.y for s in self.test_set if s.cluster_id == cluster_id]

        return X_train, y_train, X_test, y_test

    @staticmethod
    def generate_data(dataset, value='x'):
        if value == 'x':
            # return [sample.x for sample in dataset]
            for sample in dataset:
                yield sample.x
        elif value == 'y':
            for sample in dataset:
                yield sample.y
        else:
            for sample in dataset:
                yield sample.cluster_id


class Sample:

    def __init__(self, x, y, cluster_id=-1):
        self.x = x
        self.y = y
        self.cluster_id = cluster_id

    def __str__(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.cluster_id)


class Classifier:

    def __init__(self, n_estimators=10, classifier='bagging_svm'):
        self.params = {
            'C': 5,
            'kernel': 'rbf',  # , 'linear', 'rbf', 'polynomial'
            'gamma': 10,
            'degree': 1,
            'coef0': 1
        }
        self.params2 = {
            'C': 100,
            'kernel': 'rbf',
            'gamma': 0.001,
            'degree': 3,
            'coef0': 3
        }
        self.svc = SVC(**self.params, random_state=42)
        # self.svc = TSVC(**self.params2, random_state=42)
        self.n_estimators = n_estimators
        self.scaler = Normalizer()
        self.clf = None
        if classifier == 'cluster_svm':
            self.cluster_svm()
        elif classifier == 'bagging_svm':
            self.bagging_svm()
        elif classifier == 'boosting_svm':
            self.boosting_svm()

    def cluster_svm(self):
        self.clf = self.svc

    def bagging_svm(self):
        self.clf = BaggingClassifier(base_estimator=self.svc, n_estimators=self.n_estimators,
                                     max_samples=1.0, max_features=1.0, bootstrap=True,
                                     bootstrap_features=False, n_jobs=1, random_state=1)

    def boosting_svm(self):
        self.clf = AdaBoostClassifier(self.svc, n_estimators=self.n_estimators, learning_rate=1, algorithm='SAMME')

    def fit(self, X, y, is_normalize=True):
        print("######### begin training ################")
        if is_normalize:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        t_start = time.time()
        self.clf.fit(X, y)
        used_time = time.time() - t_start
        print("########## finish training ##############")
        print("used time:", str(used_time))

        return self.clf

    def predict(self, X, is_normalize=True):
        if is_normalize:
            X = self.scaler.transform(X)
        labels = self.clf.predict(X)
        return labels


if __name__ == "__main__":
    # X, y = fetch_covtype(return_X_y=True)
    X_train, y_train = load_svmlight_file('./datasets/letter.scale.tr')
    X_test, y_test = load_svmlight_file('./datasets/letter.scale.t')
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    print(X_train.shape)
    dataset = Dataset(n_clusters=10)
    dataset.process_data(X_train, y_train, X_test, y_test)
    unique_cluster_ids = set([x for x in dataset.generate_data(dataset.train_set, value='c')])
    print(unique_cluster_ids)
    y_true = []
    y_pred = []
    for cluster_id in unique_cluster_ids:
        X_train0, y_train0, X_test0, y_test0 = dataset.data_from_aspect(cluster_id=cluster_id)
        clf = Classifier(classifier='cluster_svm')
        clf.fit(X_train0, y_train0)
        labels = clf.predict(X_test0)
        y_true.extend(y_test0)
        y_pred.extend(labels)
    # X = X[:50000]
    # y = y[:50000]
    # print(X.shape)
    # dataset = Dataset(n_clusters=5)
    # X_train, y_train, X_test, y_test = dataset.process_after_split(X, y)
    # new_X, new_y, _, _ = dataset.data_from_aspect(cluster_id=0)

    # clf = Classifier(classifier='boosting_svm')
    # clf.fit(X_train, y_train)
    # labels = clf.predict(X_test)
    print(classification_report(y_true, y_pred))
