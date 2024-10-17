from ctypes import sizeof
import getopt
from math import sqrt
import math
import os
import pickle
import sys
from sklearn.cluster._kmeans import KMeans
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, SVR
from sklearn.ensemble import *
from sklearn.preprocessing import Normalizer
import numpy as np
from scipy.sparse import vstack
from sklearn.metrics import accuracy_score, confusion_matrix, explained_variance_score, f1_score, classification_report, mean_absolute_error, mean_squared_error, normalized_mutual_info_score, r2_score
from sklearn.decomposition import TruncatedSVD
from thundersvm import SVC as TSVC, SVR as TSVR
from fast_pytorch_kmeans import KMeans as fpkmeans
from FC import featureConstruction
from hyperoptall import HyperoptTunerAll
import time
import traceback
import torch
import logging
from memory_profiler import profile
from MAFC import Dataset as MafcData
import warnings


warnings.filterwarnings("ignore")
class Dataset:

    def __init__(self, **kwargs):
        '''
        '''
        self.train_set_index = {}
        self.test_set_index = {}
        self.train_labels_index = {}
        self.train_data = None
        self.test_data = None
        self.train_y = None
        self.test_y = None
        self.n_clusters = 1
        self.logger = kwargs['logger']
        if kwargs.get('n_clusters') is not None:
            self.n_clusters = kwargs['n_clusters']
        if kwargs.get('labels') is not None:
            self.labels = kwargs['labels']
        self.kmeans = KMeans(n_clusters=self.n_clusters,
                             max_iter=3000,
                             random_state=42,
                             verbose=False)

        self.ispytorchkmeans = False
        if kwargs.get('ispytorchkmeans') is not None and kwargs['ispytorchkmeans']:
            self.ispytorchkmeans = True
            self.kmeans = fpkmeans(n_clusters=self.n_clusters,
                             max_iter=3000,
                             minibatch=10000)
        self.isRegression = False
        if kwargs.get('isRegression') is not None and kwargs['isRegression']:
            self.isRegression = True
        self.lsafeatures = 0
        if kwargs.get('lsafeatures') is not None and kwargs['lsafeatures'] > 0:
            self.lsafeatures = kwargs['lsafeatures']
        self.issparse = False
        if kwargs.get('issparse') is not None and kwargs['issparse']:
            self.issparse = True
        self.kwargs = kwargs

    def load_data_from_file(self, path):
        pass

    def largedata_cluster(self, X_train, X_test):
        clusters = self.n_clusters
        if self.kwargs.get('kmeansfromfile') is not None :
            train_path = dataset_path + "kmeans/" + self.kwargs['kmeansfromfile'] + '_train' + str(self.n_clusters) + "_" + str(self.lsafeatures)
            test_path = dataset_path + "kmeans/" + self.kwargs['kmeansfromfile'] + '_test' + str(self.n_clusters)+ "_" + str(self.lsafeatures)
            self.logger.info(f"load kmeans from file: {train_path} and {test_path}.")
            if not os.path.exists(train_path + '.plk') or not os.path.exists(test_path + '.plk'):
                self.logger.warning(f"load kmeans from file: {train_path} or {test_path} not exists.")
            else:
                with open(train_path + '.plk', 'rb') as f:
                    train_cluster_ids = pickle.load(f)
                with open(test_path + '.plk', 'rb') as f:
                    test_cluster_ids = pickle.load(f)
                return train_cluster_ids, test_cluster_ids

        X_train_lsa = X_train
        X_test_lsa = X_test
        if self.lsafeatures > 0:
            t1 = time.time()
            svd = TruncatedSVD(n_components=self.lsafeatures, algorithm='arpack', n_iter=3000, random_state=42)
            lsa = make_pipeline(svd, Normalizer(copy=False))
            X_train_lsa = lsa.fit_transform(X_train_lsa)
            X_test_lsa = lsa.transform(X_test_lsa)
            t2 = time.time()
            self.logger.info(f"time for lsa: {t2-t1} s.")
            self.logger.info(f"X_train_lsa shape: {X_train_lsa.shape}, X_test_lsa shape: {X_test_lsa.shape}")

        limitsize = 2e9
        while clusters * X_train_lsa.shape[0] > limitsize: clusters //= 10
        if clusters < 1:
            self.logger.error(f"clusters is less than 1, please check the dataset size {X_train_lsa.shape}.")
            sys.exit(2)
        train_indexcluster_list = []
        test_indexcluster_list = []
        if clusters == self.n_clusters:
            train_cluster_ids = self.split_via_cluster(X_train_lsa)
            test_cluster_ids = self.cluster_predict(X_test_lsa)
        else:
            # 因为self.n_clusters太大了，所以需要多次聚类
            have_cluster = 1
            clusterid = 0
            max_cluster_id = 0
            maxinstances = 0
            tempX_train_list = [X_train_lsa]
            tempX_test_list = [X_test_lsa]
            train_indexcluster_list = [[[i, 0] for i in range(X_train_lsa.shape[0])]]
            test_indexcluster_list = [[[i, 0] for i in range(X_test_lsa.shape[0])]]
            while have_cluster < self.n_clusters:
                self.logger.debug(f"id: {clusterid}, clusters: {clusters}, have_cluster: {have_cluster}")
                have_cluster *= clusters
                datalens = len(tempX_train_list)
                for i in range(datalens): # 对每个数据集进行聚类
                    train_index = {} # {cluster_id: [index, ...]}
                    test_index = {} # {cluster_id: [index, ...]}
                    tempX_train = tempX_train_list[i] # 数据集 
                    tempX_test = tempX_test_list[i] 
                    ks = KMeans(n_clusters=clusters,
                                max_iter=3000,
                                random_state=42,
                                verbose=False) 
                    if tempX_train.shape[0] < clusters:
                        self.logger.warning(f"tempX_train.shape[0]: {tempX_train.shape[0]} is less than clusters: {clusters}.")
                        continue
                    temp_train_cluster_ids = ks.fit_predict(tempX_train)
                    temp_test_cluster_ids = ks.predict(tempX_test)
                    # 添加划分后的数据集
                    num_adddatacluster = {}
                    for j in range(len(temp_train_cluster_ids)):
                        label = temp_train_cluster_ids[j]
                        if train_index.get(label) is None:
                            train_index[label] = []
                        train_index[label].append(j)
                    for j in range(len(temp_test_cluster_ids)):
                        label = temp_test_cluster_ids[j]
                        if test_index.get(label) is None:
                            test_index[label] = []
                        test_index[label].append(j)
                    minlabel = 0
                    flag = False
                    for label in range(len(train_index)):
                        if train_index.get(label) is None or test_index.get(label) is None:
                            if not flag: minlabel += 1
                            continue
                        maxinstances = max(maxinstances, len(train_index[label]))
                        num_adddatacluster[label] = len(num_adddatacluster)
                        if not flag and label == minlabel:
                            tempX_train_list[i] = tempX_train[train_index[label]]
                            tempX_test_list[i] = tempX_test[test_index[label]]
                            flag = True
                            continue
                        datalist = tempX_train[train_index[label]]
                        tempX_train_list.append(datalist)
                        datatestlist = tempX_test[test_index[label]]
                        tempX_test_list.append(datatestlist)
                    # 添加划分后的索引
                    train_indexclusteri = train_indexcluster_list[i] # [[index, cluster_id], ...]
                    test_indexclusteri = test_indexcluster_list[i] # [[index, cluster_id], ...]
                    new_train_indexcluster_list = [[] for _ in range(len(num_adddatacluster))] # [[[index, cluster_id], ...]]
                    new_test_indexcluster_list = [[] for _ in range(len(num_adddatacluster))] # [[[index, cluster_id], ...]]
                    for j in range(len(temp_train_cluster_ids)):
                        label = temp_train_cluster_ids[j]
                        index = train_indexclusteri[j][0]
                        if label not in num_adddatacluster:
                            continue
                        labelindex = num_adddatacluster[label]
                        if labelindex == 0:
                            new_train_indexcluster_list[labelindex].append([index, i])
                        else : 
                            new_train_indexcluster_list[labelindex].append([index, max_cluster_id + labelindex])
                    train_indexcluster_list[i] = new_train_indexcluster_list[0]
                    train_indexcluster_list.extend(new_train_indexcluster_list[1:])
                    # 验证train_indexcluster_list 和 tempX_train_list的长度是否一致
                    if len(train_indexcluster_list) != len(tempX_train_list):
                        self.logger.error(f"i = {i} train_indexcluster_list: {len(train_indexcluster_list)} length is not equal to tempX_train_list: {len(tempX_train_list)}")
                        sys.exit(2)
                    for cluster, indexlist in enumerate(train_indexcluster_list):
                        if len(indexlist) != tempX_train_list[cluster].shape[0]:
                            self.logger.error(f"train_indexcluster_list[{cluster}]: {tempX_train_list[cluster].shape[0]} length is not equal to train_index[{cluster}]: {len(indexlist)}")
                            sys.exit(2)


                    for j in range(len(temp_test_cluster_ids)):
                        label = temp_test_cluster_ids[j]
                        index = test_indexclusteri[j][0]
                        if label not in num_adddatacluster:
                            continue
                        labelindex = num_adddatacluster[label]
                        if labelindex == 0:
                            new_test_indexcluster_list[labelindex].append([index, i])
                        else : 
                            new_test_indexcluster_list[labelindex].append([index, max_cluster_id + labelindex])
                    test_indexcluster_list[i] = new_test_indexcluster_list[0]
                    test_indexcluster_list.extend(new_test_indexcluster_list[1:])
                    # 验证test_indexcluster_list 和 tempX_test_list的长度是否一致
                    if len(test_indexcluster_list) != len(tempX_test_list):
                        self.logger.error(f"i = {i} test_indexcluster_list: {len(test_indexcluster_list)} length is not equal to tempX_test_list: {len(tempX_test_list)}")
                        sys.exit(2)
                    for cluster, indexlist in enumerate(test_indexcluster_list):
                        if len(indexlist) != tempX_test_list[cluster].shape[0]:
                            self.logger.error(f"test_indexcluster_list[{cluster}]: {tempX_test_list[cluster].shape[0]} length is not equal to test_index[{cluster}]: {len(indexlist)}")
                            sys.exit(2)

                    max_cluster_id += len(num_adddatacluster) - 1

                clusters = self.n_clusters // have_cluster + 1
                while clusters * maxinstances > limitsize: clusters //= 10
                clusterid += 1
            # 均匀初始化
            train_cluster_ids = np.zeros(X_train_lsa.shape[0], dtype=np.int32)
            for i in range(len(train_cluster_ids)):
                train_cluster_ids[i] = i // (X_train_lsa.shape[0] // len(train_indexcluster_list) + 1) 
            test_cluster_ids = np.zeros(X_test_lsa.shape[0], dtype=np.int32)
            for i in range(len(test_cluster_ids)):
                test_cluster_ids[i] = i // (X_test_lsa.shape[0] // len(test_indexcluster_list) + 1)
            # 重新赋值
            for i in range(len(train_indexcluster_list)):
                for index, cluster_id in train_indexcluster_list[i]:
                    train_cluster_ids[index] = cluster_id
            for i in range(len(test_indexcluster_list)):
                for index, cluster_id in test_indexcluster_list[i]:
                    test_cluster_ids[index] = cluster_id
        self.logger.info(f"train_cluster_ids shape: {train_cluster_ids.shape}, test_cluster_ids shape: {test_cluster_ids.shape}")
        
        
        if self.kwargs.get('kmeanstofile') is not None:
            train_path = dataset_path + "kmeans/" + self.kwargs['kmeanstofile'] + '_train' + str(self.n_clusters) + "_" + str(self.lsafeatures)
            test_path = dataset_path + "kmeans/" + self.kwargs['kmeanstofile'] + '_test' + str(self.n_clusters) + "_" + str(self.lsafeatures)
            self.logger.info(f"store kmeans to file: {train_path} and {test_path}.")
            with open(train_path + '.plk', 'wb') as f:
                pickle.dump(train_cluster_ids, f)
            with open(test_path + '.plk', 'wb') as f:
                pickle.dump(test_cluster_ids, f)
            with open(train_path + '.index', 'wb') as f:
                pickle.dump(train_indexcluster_list, f)
            with open(test_path + '.index', 'wb') as f:
                pickle.dump(test_indexcluster_list, f)
        
        return train_cluster_ids, test_cluster_ids
            
    def largedata_average_cluster(self, X_train, X_test):

        k = self.n_clusters
        numinstanceprecluster = math.ceil(X_train.shape[0] / k)
        testnuminstanceprecluster = math.ceil(X_test.shape[0] / k)
        train_cluster_ids = np.zeros(X_train.shape[0], dtype=np.int32)
        test_cluster_ids = np.zeros(X_test.shape[0], dtype=np.int32)
        for index in range(X_train.shape[0]):
            train_cluster_ids[index] = min(index // numinstanceprecluster, k-1)
        for index in range(X_test.shape[0]):
            test_cluster_ids[index] = min(index // testnuminstanceprecluster, k-1)

        return train_cluster_ids, test_cluster_ids
    
    def largedata_random_cluster(self, X_train, X_test):

        k = self.n_clusters
        train_cluster_ids = np.random.randint(0, k, X_train.shape[0], dtype=np.int32)
        test_cluster_ids = np.random.randint(0, k, X_test.shape[0], dtype=np.int32)

        return train_cluster_ids, test_cluster_ids

    def process_data(self, X_train, y_train, X_test, y_test):
        
        if self.kwargs.get('clustertype') is not None and self.kwargs['clustertype'] == 'random':
            train_cluster_ids, test_cluster_ids = self.largedata_random_cluster(X_train, X_test)
        elif self.kwargs.get('clustertype') is not None and self.kwargs['clustertype'] == 'average':
            train_cluster_ids, test_cluster_ids = self.largedata_average_cluster(X_train, X_test)
        else: train_cluster_ids, test_cluster_ids = self.largedata_cluster(X_train, X_test)
        # train_cluster_ids = self.split_via_cluster(X_train_lsa)
        # test_cluster_ids = self.cluster_predict(X_test_lsa)
        for i in range(len(train_cluster_ids)):
            label = train_cluster_ids[i]
            if self.train_set_index.get(label) is None:
                self.train_set_index[label] = []
            self.train_set_index[label].append(i)
        
        for i in range(len(test_cluster_ids)):
            label = test_cluster_ids[i]
            if self.test_set_index.get(label) is None:
                self.test_set_index[label] = []
            self.test_set_index[label].append(i)

        self.train_data = X_train
        self.test_data = X_test
        self.train_y = y_train
        self.test_y = y_test
        if self.isRegression:
            return
        for i, label in enumerate(self.train_y):
            if self.train_labels_index.get(label) is None:
                self.train_labels_index[label] = [(train_cluster_ids[i], i)]
            else:
                self.train_labels_index[label].append((train_cluster_ids[i], i))
        

    def process_after_split(self, X_data, y_data, test_ratio=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                            test_size=test_ratio,
                                                            random_state=0)
        self.process_data(X_train, y_train, X_test, y_test)
        return X_train, y_train, X_test, y_test

    def split_via_cluster(self, X_data):
        if self.ispytorchkmeans:
            X_data_cuda = torch.from_numpy(X_data).cuda()
            return self.kmeans.fit_predict(X_data_cuda).cpu().numpy()
        return self.kmeans.fit_predict(X_data)

    def cluster_predict(self, x_data):
        if self.ispytorchkmeans:
            x_data_cuda = torch.from_numpy(x_data).cuda()
            return self.kmeans.predict(x_data_cuda).cpu().numpy()
        return self.kmeans.predict(x_data)

    def data_from_aspect(self, cluster_id, is_sampling=True):
        X_train_add = []
        y_train_add = []
        if self.train_set_index.get(cluster_id) is None:
            X_train = np.array([])
            y_train = []
        else :
            X_train = self.train_data[self.train_set_index[cluster_id]]
            y_train = self.train_y[self.train_set_index[cluster_id]]
        if self.test_set_index.get(cluster_id) is None:
            X_test = np.array([])
            y_test = []
        else: 
            X_test = self.test_data[self.test_set_index[cluster_id]]
            y_test = self.test_y[self.test_set_index[cluster_id]]
        # self.logger.info(len(X_train))
        if is_sampling and self.isRegression is False:
            t1  = time.time()
            label_counter = Counter(y_train)
            # self.logger.info(label_counter)
            max_label, mc_label = label_counter.most_common()[0]
            need_label_count = label_counter
            if len(label_counter) / len(set(self.train_y)) < 0.7:
                need_label_count = Counter(self.train_y)
            for label in need_label_count:
                if label == max_label: continue
                if label not in label_counter:
                    label_counter[label] = 0
                label_count = label_counter[label]
                max_label_count = int(self.labels[label] / self.labels[max_label] * mc_label) 
                # self.logger.info(f"label: {label}, max_label_count: {max_label_count}, max_label: {max_label}, mc_label: {mc_label}")
                if label_count < max_label_count:
                    for cluster, index in self.train_labels_index[label]:
                        if cluster == cluster_id: continue
                        X_train_add.append(self.train_data[index])
                        y_train_add.append(self.train_y[index])
                        label_count += 1
                        if label_count >= max_label_count:
                            break
            t2 = time.time()
            self.logger.debug(f"time for sampling: {t2-t1} s.")
            t1 = time.time()
            if self.issparse:
                for x in X_train_add:
                    X_train = vstack((X_train, x))
            else:
                if len(X_train_add) > 0:
                    X_train = np.vstack((X_train, X_train_add))
            if len(y_train_add) > 0:
                y_train = np.hstack((y_train, y_train_add))
            t2 = time.time()
            self.logger.debug(f"time for add sampling to dataset: {t2-t1} s.")
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

    def reprocess_data(self, clusterid, k):

        trainindexlist = self.train_set_index[clusterid]
        testindexlist = self.test_set_index[clusterid]
        X_train = self.train_data[trainindexlist]
        y_train = self.train_y[trainindexlist]
        X_test = self.test_data[testindexlist]
        y_test = self.test_y[testindexlist]
        kmeans = KMeans(n_clusters=k, max_iter=3000, random_state=42, verbose=False)
        train_cluster_ids = kmeans.fit_predict(X_train)
        test_cluster_ids = kmeans.predict(X_test)
        # train_cluster_ids = self.split_via_cluster(X_train_lsa)
        # test_cluster_ids = self.cluster_predict(X_test_lsa)
        oldclusterset = set(self.train_set_index.keys())
        tempk = len(self.train_set_index)
        self.train_set_index[clusterid] = []
        for i in range(len(train_cluster_ids)):
            label = train_cluster_ids[i]
            if label == 0: index = clusterid
            else : index = tempk + label - 1
            if self.train_set_index.get(index) is None:
                self.train_set_index[index] = []
            self.train_set_index[index].append(trainindexlist[i])
        
        tempk = len(self.test_set_index)
        self.test_set_index[clusterid] = []
        for i in range(len(test_cluster_ids)):
            label = test_cluster_ids[i]
            if label == 0: index = clusterid
            else : index = tempk + label - 1
            if self.test_set_index.get(index) is None:
                self.test_set_index[index] = []
            self.test_set_index[index].append(testindexlist[i])
        
        if self.isRegression:
            return
        # 删除self.train_labels_index中的clusterid
        for label in self.train_labels_index:
            newlist = []
            for cluster, index in self.train_labels_index[label]:
                if cluster != clusterid:
                    newlist.append((cluster, index))
            self.train_labels_index[label] = newlist
        for i, label in enumerate(y_train):
            cluster = train_cluster_ids[i]
            if cluster == 0: cluster = clusterid
            else : cluster = tempk + cluster - 1
            if self.train_labels_index.get(label) is None:
                self.train_labels_index[label] = []
            self.train_labels_index[label].append((cluster, trainindexlist[i]))
        cluster_idset = {clusterid}
        cluster_idset = cluster_idset.union(set(self.train_set_index.keys()) - oldclusterset)
        return cluster_idset
        
        
class Sample:

    def __init__(self, x=None, y=None, index=0,  cluster_id=-1):
        self.x = x
        self.y = y
        self.index = index
        self.cluster_id = cluster_id

    def __str__(self):
        return str(self.x) + ", " + str(self.y) + str(self.index) + ", " + str(self.cluster_id)


class Classifier:

    def __init__(self, n_estimators=10, classifier='cluster_svm', maxiter=500, **kwargs):
        self.logger = kwargs['logger']
        self.params = {
            'C': 128,
            # 'kernel': 'rbf',  # , 'linear', 'rbf', 'polynomial'
            'gamma': 0.02,
            # 'degree': 1,
            # 'coef0': 1
            'decision_function_shape': 'ovo',
        }
        self.params2 = {
            'C': 128,
            'kernel': 'rbf',  # , 'linear', 'rbf', 'polynomial'
        }
        # self.cifar10_params = {'C': 10, 'kernel': 'rbf','gamma': 0.002}
        # self.log1p_params = {'C': 2000, 'gamma': 0.001, 'kernel': 'linear'}
        # self.sector_params = {'C': 432428.4244674809, 'coef0': 7.924795884923858, 'degree': 3, 'gamma': 0.00014889888096680627, 'kernel': 'polynomial'}
        if kwargs.get('svmparams') is not None:
            self.params = kwargs['svmparams']
            self.params2 = kwargs['svmparams']
        self.gpuid = 0
        if kwargs.get('gpuid') is not None:
            self.gpuid = kwargs['gpuid']
        self.isverbose = False
        if kwargs.get('isverbose') is not None and kwargs['isverbose']:
            self.isverbose = True
        self.svmtype = 'thundersvm'
        if kwargs.get('svmtype') is not None:
            self.svmtype = kwargs['svmtype']
        
        if kwargs.get('isRegression') is not None and kwargs['isRegression']:
            if self.svmtype == 'libsvm':
                self.model = SVR(**self.params2, random_state=42, max_iter=maxiter)
            else : 
                self.model = TSVR(**self.params2, gpu_id=self.gpuid, verbose=self.isverbose, max_iter=maxiter)

        else :
            if self.svmtype == 'libsvm':
                self.model = SVC(**self.params, random_state=42, max_iter=maxiter)
            else :
                self.model = TSVC(**self.params, random_state=42, gpu_id=self.gpuid, verbose=self.isverbose, max_iter=maxiter)
        
        self.n_estimators = n_estimators
        self.scaler = Normalizer()
        self.clf = self.model
        if classifier == 'cluster_svm':
            self.cluster_svm()
        elif classifier == 'bagging_svm':
            self.bagging_svm()
        elif classifier == 'boosting_svm':
            self.boosting_svm()

    def cluster_svm(self):
        self.clf = self.model

    def bagging_svm(self):
        self.clf = BaggingClassifier(base_estimator=self.model, n_estimators=self.n_estimators,
                                     max_samples=1.0, max_features=1.0, bootstrap=True,
                                     bootstrap_features=False, n_jobs=1, random_state=1)

    def boosting_svm(self):
        self.clf = AdaBoostClassifier(self.model, n_estimators=self.n_estimators, learning_rate=1, algorithm='SAMME')

    def fit(self, X, y, is_normalize=True):
        ("######### begin training ################")
        if is_normalize:
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        t_start = time.time()
        self.clf.fit(X, y)
        used_time = time.time() - t_start
        # self.logger.info("########## finish training ##############")
        # self.logger.info("used time:", str(used_time))

        return self.clf, used_time

    def predict(self, X, is_normalize=True):
        if is_normalize:
            X = self.scaler.transform(X)
        labels = self.clf.predict(X)
        return labels


class LSVMPipeline:

    def __init__(self, dataset_path, datasets_dict, dataset_name_list=[], dataset_notest_name=[], n_clusters_list={}, **kwargs):
        self.dataname = None
        self.dataset_name = None
        self.theta = 0.025
        if kwargs.get('dataname') is not None:
            self.dataname = [kwargs['dataname']]
        self.dataset_path = dataset_path
        self.datasets_dict = datasets_dict
        self.dataset_name_list = dataset_name_list
        self.dataset_notest_name = dataset_notest_name
        self.n_clusters_list = n_clusters_list
        self.rate = 0.2
        if kwargs.get('rate') is not None:
            self.rate = kwargs['rate']
        self.kwargs = kwargs
        self.loggerlevel = logging.INFO
        if kwargs.get('loggerlevel') is not None:
            self.loggerlevel = kwargs['loggerlevel']
        self.logger = logging.getLogger('LSVMPipeline')
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(self.loggerlevel)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.islogging = False
        if kwargs.get('islogging') is not None:
            self.islogging = True
            if kwargs['islogging'] == "default":
                fh = logging.FileHandler('LSVMPipeline.log')
            else :
                fh = logging.FileHandler(kwargs['islogging'])
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        
    
    def modify_params(self, xdata, ydata):
        m, n = xdata.shape
        databyte = xdata.dtype.itemsize
        classes = 1
        if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
            self.kwargs['is_sapmle'] = False
        else: classes = len(set(ydata))
        # if m * n * databyte > 1e10:
        #     self.kwargs['issparse'] = True
        if m * classes * (classes - 1) /2 * databyte > 2e9:
            self.kwargs['svmtype'] = 'libsvm'
        if n < 100:
            self.kwargs['lsafeatures'] = 0
        classes = len(set(ydata))
        if classes >= 10:
            self.kwargs['isbanlance'] = True
        self.logger.info(f"modify_params: issparse: {self.kwargs['issparse']}, svmtype: {self.kwargs['svmtype']}, lsafeatures: {self.kwargs['lsafeatures']}, isbanlance: {self.kwargs['isbanlance']}")

    def getLabelScores(self, y_test, labels, training_error = -1):
        # self.logger.info(f"classification_report:\n{classification_report(y_test, labels)}")
        self.logger.info(f"confusion matrix:\n {confusion_matrix(y_test, labels, labels=list(Counter(y_test).keys()))}")
        self.logger.info(f"acc: {accuracy_score(y_test, labels)}")
        self.logger.info(f"f1:  {f1_score(y_test, labels, average='macro')}")
        if training_error != -1:
            self.logger.info(f"training_acc: {training_error}")

    def getRegressionScores(self, y_test, y_pred, training_error = -1):

        if self.kwargs.get('isNormalization') is not None and self.kwargs['isNormalization']:
            miny = min(y_test)
            maxy = max(y_test)
            y_test = (y_test - miny) / (maxy - miny)
            y_pred = (y_pred - miny) / (maxy - miny)
        self.logger.info(f"RMSE: {sqrt(mean_squared_error(y_test, y_pred))}")
        self.logger.info(f"MSE: {mean_squared_error(y_test, y_pred)}")
        self.logger.info(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        self.logger.info(f"varince: {explained_variance_score(y_test, y_pred)}")
        self.logger.info(f"r2: {r2_score(y_test, y_pred)}")
        if training_error != -1:
            self.logger.info(f"train mean_absolute_error: {training_error}")

    def create_dataset(self, X_train, y_train, X_test, y_test):
        t1 = time.time()
        if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
            dataset = Dataset(logger=self.logger, **self.kwargs)
        else: 
            labels = Counter(y_train)
            dataset = Dataset(labels = labels, logger=self.logger, **self.kwargs)  
        dataset.process_data(X_train, y_train, X_test, y_test)
        t2 = time.time()
        self.logger.info(f"time for clustering: { t2-t1} s.")
        return dataset

    # @profile
    def run2(self, dataset):
        # total_time = 0
        train_time = 0
        train_pred_time = 0
        test_pred_time = 0
        t1 = time.time()
        unique_cluster_ids = set(dataset.train_set_index.keys())
        self.logger.info(f"clusters: {len(unique_cluster_ids)}")
        y_true = []
        y_pred = []
        y_train_true = []
        y_pred_true = []
        # best_sample_seq = [0 for x in unique_cluster_ids]
        # best_pred_labels = [0 for x in unique_cluster_ids]
        flag = True
        while len(unique_cluster_ids):
            # best_acc = -1
            # best_f1 = -1
            # best_train_pred = None
            # best_test_pred = None
            # best_train_true = None
            cluster_id = unique_cluster_ids.pop()
            # if cluster_id == 0 and flag and self.dataset_name == "sector":
                # flag = False
                # newclusterset = dataset.reprocess_data(cluster_id, 2)
                # unique_cluster_ids = unique_cluster_ids.union(newclusterset)
                # continue
            if cluster_id == 1:
                continue
            self.logger.debug(f"cluster_id: {cluster_id} begin.")
            is_sampling = False
            if self.kwargs.get('isbanlance') is not None and self.kwargs['isbanlance']:
                is_sampling = True
            X_train0, y_train0, X_test0, y_test0 = dataset.data_from_aspect(cluster_id=cluster_id, is_sampling=is_sampling)
            if X_train0.shape[0] <= 0 or X_test0.shape[0] <= 0 or len(set(y_train0)) <= 1 :
                continue
            


            def early_stop_fn(iteration_stop_count=5, percent_increase=0.0):
                def stop_fn(trials, best_loss=None, iteration_no_progress=0):
                    new_loss = trials.trials[len(trials.trials)-1]['result']['loss']
                    if best_loss is None:
                        return False, [new_loss, iteration_no_progress+1]
                    if 1.0 - new_loss > bestaccandf1[self.dataset_name][0] + self.theta:
                        return True, [best_loss, iteration_no_progress]

                    best_loss_threshold = best_loss - abs(best_loss * (percent_increase/100.0))
                    if new_loss < best_loss_threshold:
                        best_loss = new_loss
                        iteration_no_progress = 0
                    else:
                        iteration_no_progress += 1
                        # self.logger.debug("No progress made: %d iteration on %d. best_loss=%.2f, best_loss_threshold=%.2f, new_loss=%.2f"
                        #             % (iteration_no_progress, iteration_stop_count, best_loss, best_loss_threshold, new_loss))

                    return iteration_no_progress >= iteration_stop_count, [best_loss, iteration_no_progress]
                
                return stop_fn


            def filterFeature(X_train, X_test, y_train0, igrate):
                selectlist = []
                for i in range(len(X_train[0])):
                    if normalized_mutual_info_score(X_train[:,i], y_train0) > igrate:
                        selectlist.append(i)
                X_train = X_train[:,selectlist]
                X_test = X_test[:,selectlist]
                return X_train, X_test

            if self.kwargs.get('Filter') is not None and self.kwargs['Filter']:
                igrate = self.kwargs.get('igrate')
                X_train0, X_test0 = filterFeature(X_train0, X_test0, y_train0, igrate=igrate)

            if self.kwargs.get('autofc') is not None and self.kwargs['autofc']:
                mafcdata = MafcData(name = self.dataset_name) 
                mafcdata.fromArray(X_train0)
                fctraindata = featureConstruction(mafcdata)
                
                mafcdata.fromArray(X_test0)
                fctestdata = featureConstruction(mafcdata)
                print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
                fctraindataarray = np.array(fctraindata)
                fctestdataarray = np.array(fctestdata)
                fctraindataarray = np.nan_to_num(fctraindataarray)
                fctestdataarray = np.nan_to_num(fctestdataarray)
                
                scaler2 = Normalizer().fit(fctraindataarray)
                fctraindataarray = scaler2.transform(fctraindataarray)
                fctestdataarray = scaler2.transform(fctestdataarray)
                # X_train0, X_test0 = filterFeature(fctraindataarray, fctestdataarray, y_train0, igrate=igrate)
            
            if self.kwargs.get('selection') is not None and self.kwargs['selection']:
                pass
            
            if self.kwargs.get('hyoptparams') is not None:
                hyoptparams = self.kwargs['hyoptparams']
                htopt = HyperoptTunerAll(X_train0, y_train0, X_test0, y_test0, maxiter=100)
                htopt.gpu_id = self.kwargs.get('gpuid')
                if self.kwargs.get('svmtype') is not None :
                    htopt.modeltype = self.kwargs['svmtype']
                htopt.tune_params(hyoptparams['maxiter'], maxtimehours=hyoptparams['maxhours'], init_params=hyoptparams.get('params'), early_stop_fn=early_stop_fn(hyoptparams['maxrepeat']))
                self.kwargs['svmparams'] = htopt.best_cfg
                self.logger.debug(f"best params: {htopt.best_cfg}")
            clf = Classifier(logger=self.logger, **self.kwargs)
            self.logger.debug(f"cluster_id: {cluster_id} data set shape: {X_train0.shape} {X_test0.shape}.")
            t1 = time.time()
            _, used_time = clf.fit(X_train0, y_train0)
            traint1 = time.time()
            train_time += traint1 - t1
            self.logger.debug(f"cluster_id: {cluster_id} train time: {traint1 - t1} s.")
            # train_labels = clf.predict(X_train0)
            train_labels = y_train0
            train_pred_time += time.time() - traint1
            self.logger.debug(f"cluster_id: {cluster_id} train_pred time: {time.time() - traint1} s.")
            testt1 = time.time()
            labels = clf.predict(X_test0)
            test_pred_time += time.time() - testt1
            self.logger.debug(f"cluster_id: {cluster_id} test_pred time: {time.time() - testt1} s.")
            if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
                training_error = mean_absolute_error(y_train0, train_labels)
                self.logger.debug(f"cluster_id: {cluster_id}: train error: {training_error}")
            else: self.logger.debug(f"{cluster_id}: acc: {accuracy_score(y_test0, labels)}, f1: {f1_score(y_test0, labels, average='macro')}") 
            y_true.extend(y_test0)
            y_pred.extend(labels)
            y_train_true.extend(y_train0)
            y_pred_true.extend(train_labels)
            # 手动释放内存
            del X_train0, y_train0, X_test0, y_test0
        t2 = time.time()
        self.logger.info(f"time for training: { train_time} s.")
        self.logger.info(f"time for predicting: {test_pred_time} s.")
        self.logger.info(f"time for predicting training data: {train_pred_time} s.")
        # total_time += t2-t1
        self.logger.info(f"total time: { t2 - t1} s.",)

        return (y_true, y_pred), (y_train_true, y_pred_true), []


    def singlesvm(self, X_train, y_train, X_test, y_test):
        clf = Classifier(logger=self.logger, **self.kwargs)
        self.logger.info("#################### single svm ###################\n")
        start = time.time()#
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        self.logger.info(f"time for training: {t2-t1} s\n")
        t1 = time.time()
        labels = clf.predict(X_test)
        t2 = time.time()
        self.logger.info(f"time for predicting: {t2-t1} s\n")
        t1 = time.time()
        # if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
        #     training_error = mean_absolute_error(y_train, clf.predict(X_train))
        # else: training_error = accuracy_score(y_train, clf.predict(X_train))
        training_error = 1
        t2 = time.time()
        self.logger.info(f"time for predicting training data: {t2-t1} s\n")
        end = time.time()
        self.logger.info(f"total time: {end-start} s\n")
        if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
            self.getRegressionScores(y_test, labels, training_error)
        else: self.getLabelScores(y_test, labels, training_error)
        self.logger.info(f"svm params: {self.kwargs['svmparams']}\n")


    def ourspieline(self, X_train, y_train, X_test, y_test):
        self.logger.info("#################### ourspieline ####################\n")
        dataset = self.create_dataset(X_train, y_train, X_test, y_test)
        test_result, train_result, best_seq = self.run2(dataset)
        if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
            training_error = mean_absolute_error(train_result[0], train_result[1])
            self.getRegressionScores(test_result[0], test_result[1], training_error)
        else:
            training_error = accuracy_score(train_result[0], train_result[1])
            self.getLabelScores(test_result[0], test_result[1], training_error)
        # self.logger.info("best_sample_seq: ",best_seq)

    def tune_params(self, X_train, y_train, X_test, y_test):
        hyoptparams = self.kwargs['hyopt']
        ht = HyperoptTunerAll(X_train, y_train, X_test, y_test)
        if self.kwargs.get('svmtype') is not None :
            ht.modeltype = self.kwargs['svmtype']
            ht.gpu_id = self.kwargs.get('gpuid')
            ht.tune_params(hyoptparams['maxiter'], maxtimehours=hyoptparams['maxhours'], init_params=hyoptparams.get('params'))
        self.logger.info(f"best params: {ht.best_cfg}")
        self.logger.info(f"best acc: {ht.best_acc}")
        self.logger.info(f"best f1: {ht.best_f1}")
        self.kwargs['svmparams'] = ht.best_cfg
    
    def calcclustertime(self):
        if self.dataname is not None:
            self.dataset_name_list = self.dataname 
        for dataset_name in self.dataset_name_list:
            self.kwargs['kmeanstofile'] = dataset_name
            self.kwargs['kmeansfromfile'] = dataset_name
            t1 = time.time()
            if dataset_name in self.dataset_notest_name:
                x_all, y_all = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][0], n_features=self.datasets_dict[dataset_name][2])
                self.modify_params(x_all, y_all)
                if self.kwargs.get('issparse') is not None and self.kwargs['issparse']:
                    pass
                else :x_all = x_all.toarray()
                X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=self.rate, random_state=42)
            else :
                X_train, y_train = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][0], n_features=self.datasets_dict[dataset_name][2])
                X_test, y_test = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][1], n_features=self.datasets_dict[dataset_name][2])
                self.modify_params(X_train, y_train)
                if self.kwargs.get('issparse') is not None and self.kwargs['issparse']:
                    pass
                else:
                    X_train = X_train.toarray()
                    X_test = X_test.toarray()
            t2 = time.time()
            self.logger.info(f"time for loading data: {t2-t1} s.")
            self.logger.info(f"dataset name: {dataset_name}, traindata shape: {X_train.shape}, testdata shape: {X_test.shape}\n")
            dataset = self.create_dataset(X_train, y_train, X_test, y_test)

    # @profile
    def run(self):
        if self.dataname is not None:
            self.dataset_name_list = self.dataname 
        for dataset_name in self.dataset_name_list:
            isourstemp = self.kwargs.get('isours')
            self.dataset_name = dataset_name
            if dataset_name == 'avazu':
                self.kwargs['isours'] = True
            
            self.kwargs['kmeanstofile'] = dataset_name
            self.kwargs['kmeansfromfile'] = dataset_name
            t1 = time.time()
            if dataset_name in self.dataset_notest_name:
                x_all, y_all = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][0], n_features=self.datasets_dict[dataset_name][2])
                self.modify_params(x_all, y_all)
                if self.kwargs.get('issparse') is not None and self.kwargs['issparse']:
                    pass
                else :x_all = x_all.toarray()
                X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=self.rate, random_state=42)
            else :
                X_train, y_train = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][0], n_features=self.datasets_dict[dataset_name][2])
                X_test, y_test = load_svmlight_file(self.dataset_path + self.datasets_dict[dataset_name][1], n_features=self.datasets_dict[dataset_name][2])
                self.modify_params(X_train, y_train)
                if self.kwargs.get('issparse') is not None and self.kwargs['issparse']:
                    pass
                else:
                    X_train = X_train.toarray()
                    X_test = X_test.toarray()
            t2 = time.time()
            self.logger.info(f"time for loading data: {t2-t1} s.")
            self.logger.info(f"dataset name: {dataset_name}, traindata shape: {X_train.shape}, testdata shape: {X_test.shape}\n")
            if self.kwargs.get('hyopt') is not None:
                self.tune_params(X_train, y_train, X_test, y_test)
                self.logger.info(f'run over \n')
                self.kwargs['isours'] = isourstemp
                self.logger.info(f"dataset: {dataset_name}; kwargs: {self.kwargs}")
                continue
            if not self.kwargs['isours']:
                try:
                    self.singlesvm(X_train, y_train, X_test, y_test)
                except Exception as e:
                    self.logger.error(f"traceback: {traceback.format_exc()}")
                    self.logger.error(f"singlesvm error: {e}")
            if self.dataname is None and n_clusters_list.get(dataset_name) is not None:
                self.kwargs['n_clusters'] = n_clusters_list[dataset_name]
            if self.kwargs['n_clusters'] == 1:
                self.logger.info(f"we suggest n_clusters > 1, but {self.kwargs['n_clusters']} given.")
            try:
                self.ourspieline(X_train, y_train, X_test, y_test)
                if self.kwargs.get('isRegression') is not None and self.kwargs['isRegression']:
                    pass
                elif self.kwargs.get('isbanlance') is not None and self.kwargs['isbanlance'] is False:
                    self.kwargs['isbanlance'] = not self.kwargs['isbanlance']
                    self.ourspieline(X_train, y_train, X_test, y_test)
            except Exception as e:
                self.logger.error(f"traceback: {traceback.format_exc()}")
                self.logger.error(f"ourspieline error: {e}")
            self.logger.info(f'run over \n')
            self.kwargs['isours'] = isourstemp
            self.logger.info(f"dataset: {dataset_name}; kwargs: {self.kwargs}")





if __name__ == "__main__":
    # 定义短选项和长选项
    short_opts = "r:d:n:g:RvOs:l:pbhe:L:SP:t:k:K:y:c:Y:CfENi:F"
    long_opts = ["root_path", "datasetname", "n_clusters", "gpuid", "Regression", "verbose", \
                 "Ours", "svmtype", "lsa", "pytorchkmeans", "banlance", "help", "loggerlevel", \
                 "islogging", "issparse", "svmparams", "rate", "kmeanstofile", "kmeansfromfile",\
                 "hyopt", "clustertype", "hyoptparams", "Clustertime", "autofc", "selection", \
                 "isNormalization", "igrate", "Filter"]

    # 解析命令行参数l:
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
    except getopt.GetoptError:
        print("Invalid command line arguments")
        sys.exit(2)
    root_path = "/home/Zhengzy/code/absa-svm/"
    dataset_path = root_path + "datasets/data/"
    params = {'n_clusters': 10, 'issparse': False, 'svmtype': 'thundersvm', 'rate': 0.2, 'isRegression': False, \
              'isverbose': False, 'isours': False, 'lsafeatures': 10, 'ispytorchkmeans': False, 'isbanlance': False, \
              'loggerlevel': logging.INFO, 'islogging': None, 'svmparams': None, 'gpuid': 0, 'kmeanstofile': 'defaultdata', \
              'kmeansfromfile':'defaultdata', 'igrate': 0.01}
    
    
    for opt, arg in opts:
        if opt in ("-r", "--root_path"):
            root_path = arg
        elif opt in ("-d", "--datasetname"):
            params['dataname'] = arg
        elif opt in ("-n", "--n_clusters"):
            params['n_clusters'] = int(arg)
        elif opt in ("-g", "--gpuid"):
            params['gpuid'] = int(arg)
        elif opt in ("-R", "--Regression"):
            params['isRegression'] = True
        elif opt in ("-v", "--verbose"):
            params['isverbose'] = True
        elif opt in ("-O", "--Ours"):
            params['isours'] = True
        elif opt in ("-s", "--svmtype"):
            params['svmtype'] = arg
        elif opt in ("-l", "--lsa"):
            params['lsafeatures'] = int(arg)
        elif opt in ("-p", "--pytorchkmeans"):
            params['ispytorchkmeans'] = True
        elif opt in ("-b", "--banlance"):
            params['isbanlance'] = True
        elif opt in ("-e", "--loggerlevel"):
            params['loggerlevel'] = int(arg)
        elif opt in ("-L", "--islogging"):
            params['islogging'] = arg
        elif opt in ("-S", "--issparse"):
            params['issparse'] = True
        elif opt in ("-P", "--svmparams"):
            params['svmparams'] = eval(arg)
        elif opt in ("-t", "--rate"):
            params['rate'] = float(arg)
        elif opt in ("-k", "--kmeanstofile"):
            params['kmeanstofile'] = arg
        elif opt in ("-K", "--Kmeansfromfile"):
            params['kmeansfromfile'] = arg
        elif opt in ("-y", "--hyopt"):
            params['hyopt'] = eval(arg)
        elif opt in ("-c", "--clustertype"):
            params['clustertype'] = arg
        elif opt in ("-Y", "--hyoptparams"):
            params['hyoptparams'] = eval(arg)
        elif opt in ("-C", "--Clustertime"):
            params['Clustertime'] = True
        elif opt in ("-f", "--autofc"):
            params['autofc'] = True
        elif opt in ("-E", "--selection"):
            params['selection'] = True
        elif opt in ("-N", "--isNormalization"):
            params['isNormalization'] = True
        elif opt in ("-i", "--igrate"):
            params['igrate'] = float(arg)
        elif opt in ("-F", "--Filter"):
            params['Filter'] = True
        elif opt in ("-h", "--help"):
            print("Usage: python whole_pipeline_ours.py [options]")
            print("-h --help: help")
            print("-r --root_path: root path of the project")
            print("-d --datasetname: name of the dataset")
            print("-n --n_clusters: number of clusters")
            print("-g --gpuid: gpu id")
            print("-R --Regression: is regression")
            print("-v --verbose: is verbose")
            print("-O --Ours: is ours")
            print("-s --svmtype: type of svm, libsvm or thundersvm")
            print("-l --lsa: number of features after lsa")
            print("-p --pytorchkmeans: use pytorchkmeans")
            print("-b --banlance: banlance the number of samples in each cluster")
            print("-e --loggerlevel: logger level")
            print("-L --islogging: is logging")
            print("-S --issparse: is sparse")
            print("-P --svmparams: svm params")
            print("-t --rate: test rate")
            print("-k --kmeanstofile: kmeans to file")
            print("-K --Kmeansfromfile: kmeans from file")
            print("-y --hyopt: hyperopt")
            print("-c --clustertype: cluster type, random or average")
            print("-Y --hyoptparams: hyperopt params {'maxiter': 20, 'maxhours': 1, 'params': [{'C': 10, 'kernel': 2,'gamma': 0.002, , 'degree': 2, 'coef0': 0, 'decision_function_shape': 0}], 'maxrepeat': 3}")
            print("-C --Clustertime: calc cluster time")
            print("-f --autofc: auto feature construction")
            print("-E --selection: feature selection")
            print("-N --isNormalization: is normalization")
            print("-i --igrate: information gain rate")
            print("-F --Filter: filter")
            sys.exit()

    datasets_dict = {'cod-rna':['cod-rna', 'cod-rna.t', 8], 'a7a':['a7a', 'a7a.t', 123], \
                     'letter':['letter.scale.tr', 'letter.scale.t', 16], \
                     'pendigits':['pendigits', 'pendigits.t', 16], 'higgs':["HIGGS", "HIGGS", 28], 
                     'url': ['url_combined', 'url_combined', 3231961], 'a9a': ['a9a', 'a9a.t', 123], \
                     'aloi': ['aloi.scale', 'aloi.scale', 128], 'sector': ['sector.scale', 'sector.t.scale', 55197],\
                     'cifar10': ['cifar10', 'cifar10.t', 3072], 'avazu': ['avazu-app', 'avazu-app.t', 1000000], \
                     'epsilon': ['epsilon_normalized', 'epsilon_normalized.t', 2000], \
                     'log1p.E2006': ['log1p.E2006.train', 'log1p.E2006.test', 4272227], \
                     'YearPredictionMSD': ['YearPredictionMSD', 'YearPredictionMSD.t', 90], \
                     'cadata': ['cadata', 'cadata', 8],\
                     }
    bestaccandf1 = {'higgs': [68.06, 67.20], 'epsilon': [89.77, 89.77], 'sector': [91.89, 91.73],'aloi': [91.79, 91.68 ], 'a9a': [84.97, 77.52], 'cifar10': [43.39, 43.03]}
    dataset_name_list = ['epsilon', 'sector', 'aloi', 'higgs', 'avazu']
    # dataset_name_list = ['epsilon', 'avazu', 'higgs']
    # dataset_name_list = ['log1p.E2006']#, 'YearPredictionMSD', 'cadata']
    dataset_notest_name = ['higgs', 'url', 'aloi', 'cadata']
    n_clusters_list = {'a9a':10, 'sector':10, 'cifar10':10, 'aloi':10, 'avazu':100000, 'higgs':100, 'epsilon':10, 'log1p.E2006':250, 'YearPredictionMSD':10, 'cadata':10}

    lsvmpipeline = LSVMPipeline(dataset_path, datasets_dict, dataset_name_list, dataset_notest_name, n_clusters_list, **params)
    if params.get('Clustertime') is not None and params['Clustertime']:
        lsvmpipeline.calcclustertime()
    else: lsvmpipeline.run()

    



