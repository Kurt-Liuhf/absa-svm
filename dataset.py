import copy
import math

import numpy as np
from file_utils import *
from stanford_nlp import StanfordNLP
import pickle
from cluster_utils import AspectCluster, WordsCluster, VectorCluster
from chi import CHI
import time


def preprocessing(data, aspectword='##'):
    nlp_helper = StanfordNLP(aspectword=aspectword)
    for sample in data:
        # 1. tokenize && pos tagging
        sample.words, sample.pos_tags = nlp_helper.pos_tag(sample.text)
        # 2. get aspect-dependent words
        aspect_term = sample.aspect.split(' ')[-1]
        tmp_text = str.replace(sample.text, aspectword, aspect_term)
        sample.dependent_words, sample.dependent_pos_tags, _ = nlp_helper.get_dependent_words(sample.words, sample.pos_tags, tmp_text, n=3, window_size=5)
        # print(sample)


def aspect_cluster(dataset, n_clusters=20, splitaspect=-1, currentaspect=0):
    ac = AspectCluster(dataset, n_clusters)
    ac.setaspect(splitaspect, currentaspect)
    _, vectors = ac.fit()
    ac.predict()
    # ac.save_cluster_result()

    return ac, vectors


def word_cluster(dataset, n_clusters=20):
    wc = WordsCluster(dataset, n_clusters)
    wc.generate_vector()


def vector_cluster(dataset, n_clusters=20):
    vc = VectorCluster(dataset, n_clusters)
    vc.fit()
    vc.predict()

def chi_calculation(dataset, ratio):
    stopwords = stop_words()
    chi_cal = CHI([" ".join(s.words) for s in dataset.train_data],
              [s.aspect_cluster for s in dataset.train_data],
              stop_words())

    chi_dict = {}
    for aspect_cluster, feature_list in chi_cal.chi_dict.items():
        chi_dict[aspect_cluster] = feature_list[0: int(len(feature_list) * ratio)]

    for sample in dataset.train_data:
        sample.bow_words = []
        sample.bow_tags = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                sample.bow_words.append(w)
                sample.bow_tags.append(w)

    for sample in dataset.test_data:
        sample.bow_words = []
        sample.bow_tags = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                sample.bow_words.append(w)
                sample.bow_tags.append(w)


class Dataset(object):
    def __init__(self, base_dir, is_preprocessed, ratio=0.3, aspectcluster=20, iscustomized=False, isgeneral=False, israndom=False, aspectword='##'):
        
        self.base_dir = base_dir
        self.aspectword = aspectword
        if isgeneral:
            if not is_preprocessed:
                pass
        if iscustomized:
            return
        self.aspect_cluster = aspectcluster
        if not is_preprocessed:
            training_path = os.path.join(base_dir, 'train.txt')
            test_path = os.path.join(base_dir, 'test.txt')
            self.train_data = self.load_raw_data(training_path)
            self.test_data = self.load_raw_data(test_path)
            preprocessing(self.train_data)
            preprocessing(self.test_data)
            if israndom:
                self.randomsplit()
            else:
                aspect_cluster(self, aspectcluster)
            word_cluster(self, 30)
            
            self.save_as_pickle()
            self.save_as_txt()
            # self.save_as_tmp()
        else:
            
            training_path = os.path.join(base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_train.plk')
            test_path = os.path.join(base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_test.plk')
            self.load_preprocessed_data(training_path, test_path)
            chi_calculation(self, ratio)

    def randomsplit(self):
        aspectcluster = self.aspect_cluster 
        for i in range(len(self.train_data)):
            self.train_data[i].aspect_cluster = np.random.randint(0, aspectcluster)
        for i in range(len(self.test_data)):
            self.test_data[i].aspect_cluster = np.random.randint(0, aspectcluster)
        # self.save_as_tmp()

    def reSplit(self, testnum = 15):

        self.numsvec = self.getNumsVec(self.test_data)
        currentnum = self.aspect_cluster
        for i in range(len(self.numsvec)):
            if self.numsvec[i] >= 2 * testnum:
                renum = self.numsvec[i] // testnum
                subdataset = self.subDataset(i, renum)
                ac, _ = aspect_cluster(subdataset, renum, i, currentnum)
                renum = ac.n_clusters
                print(f"{self.base_dir} aspectid: {i} reSplit from {currentnum} to {currentnum + renum - 2}")
                word_cluster(subdataset, self.aspect_cluster)
                tempnumsvec = self.getNumsVec(self.test_data)
                
                currentnum += renum - 1
        self.aspect_cluster = currentnum
        self.save_as_pickle()
        self.save_as_txt()
        # self.save_as_tmp()
    
    def reSplitfromvec(self, vector = [], testnum = 10):
        currentnum = self.aspect_cluster
        self.numsvec = self.getNumsVec(self.test_data)
        for i in range(len(vector)):
            aspectid = vector[i]
            renum = math.ceil(self.numsvec[aspectid] / 10)  
            subdataset = self.subDataset(aspectid, testnum)
            ac, _ = aspect_cluster(subdataset, renum, aspectid, currentnum)
            renum = ac.n_clusters
            print(f"{self.base_dir} aspectid: {aspectid} reSplit from {currentnum} to {currentnum + renum - 2}")
            word_cluster(subdataset, self.aspect_cluster)
            tempnumsvec = self.getNumsVec(self.test_data)
            currentnum += renum - 1
        self.aspect_cluster = currentnum
        self.save_as_pickle()
        self.save_as_txt() 
    
    def subDataset(self, aspectid, aspectcluster):
        traindata = []
        testdata = []
        for i in range(len(self.train_data)):
            if self.train_data[i].aspect_cluster == aspectid:
                traindata.append(self.train_data[i])
        for i in range(len(self.test_data)):
            if self.test_data[i].aspect_cluster == aspectid:
                testdata.append(self.test_data[i])
        subdata = Dataset(self.base_dir, True, iscustomized=True)
        subdata.aspect_cluster = aspectcluster
        subdata.train_data = traindata
        subdata.test_data = testdata
        return subdata
            
    
    def load_raw_data(self, path):
        data = []
        lines = read_as_list(path)
        for i in range(len(lines) // 3):
            data.append(Sample(lines[i * 3], lines[i * 3 + 1], int(lines[i * 3 + 2])))
            if self.aspectword in data[i].text:
                data[i].text = data[i].text.replace(self.aspectword, '##')
        return data

    def getNumsVec(self, data):
        numsvec = [0 for _ in range(self.aspect_cluster)]
        for sap in data:
            if sap.aspect_cluster >= 0 and sap.aspect_cluster < self.aspect_cluster:
                numsvec[sap.aspect_cluster] += 1
        sum = np.sum(numsvec)
        return numsvec
    
    

    def load_preprocessed_data(self, training_path, test_path):
        self.train_data = pickle.load(open(training_path, 'rb'))
        # start_time = time.time()
        self.test_data = pickle.load(open(test_path, 'rb'))
        # end_time = time.time()
        # self.load_test_data_time = end_time - start_time

    def save_as_pickle(self):
        plkdir = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster))
        training_path = os.path.join(plkdir, 'parsed_train.plk')
        test_path = os.path.join(plkdir, 'parsed_test.plk')
        if os.path.isdir(plkdir) is False: 
            make_dirs(plkdir)
        pickle.dump(self.train_data, open(training_path, 'wb'))
        pickle.dump(self.test_data, open(test_path, 'wb'))

    def save_as_txt(self):
        training_path = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_train.txt')
        test_path = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_test.txt')
        with open(training_path, 'w') as f:
            for sample in self.train_data:
                f.write(sample.__str__())

        with open(test_path, 'w') as f:
            for sample in self.test_data:
                f.write(sample.__str__())

    def data_from_aspect(self, aspect_cluster, is_sampling=True):
        pos = 0
        neg = 0
        net = 0
        train_samples = []
        for s in self.train_data:
            if s.aspect_cluster == aspect_cluster:
                if s.polarity == 1:
                    pos += 1
                elif s.polarity == 0:
                    net += 1
                else:
                    neg += 1
                train_samples.append(s)
        
        if is_sampling:
            # if pos == 0:
            #     maxnum = max(pos, net, neg, 1)
            #     for s in self.train_data:
            #         if s.polarity == 1 and s.aspect_cluster != aspect_cluster:
            #             train_samples.append(s)
            #             pos += 1
            #         if pos >= maxnum:
            #             break
            if net < pos:
                for s in self.train_data:
                    if s.polarity == 0 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        net += 1
                    if net >= pos:
                        break
            if neg < pos:
                for s in self.train_data:
                    if s.polarity == -1 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        neg += 1
                    if neg >= pos:
                        break
        start_time = time.time()
        test_samples = [s for s in self.test_data if s.aspect_cluster == aspect_cluster]
        end_time = time.time()

        # return train_samples, test_samples, end_time - start_time
        return train_samples, test_samples

    def get_aspect_labels(self):
        return list(set([s.aspect_cluster for s in self.train_data]))

    def save_as_tmp(self):
        cluster_path = os.path.join(self.base_dir, "aspect_cluster", str(self.aspect_cluster))
        if os.path.isdir(cluster_path) is False: 
            make_dirs(cluster_path)
        train_path = os.path.join(cluster_path, 'train')
        test_path = os.path.join(cluster_path, 'test')
        if os.path.isdir(train_path) is False: 
            make_dirs(train_path)
        if os.path.isdir(test_path) is False: 
            make_dirs(test_path)
        for s in self.train_data:
            with open(os.path.join(train_path, str(s.aspect_cluster)), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")
        for s in self.test_data:
            with open(os.path.join(test_path, str(s.aspect_cluster)), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")


class DatasetV2(object):
    def __init__(self, base_dir, is_preprocessed, ratio=0.3, aspectcluster=20, iscustomized=False, isgeneral=False,
                 israndom=False, aspectword='##'):

        self.base_dir = base_dir
        self.aspectword = aspectword
        if isgeneral:
            if not is_preprocessed:
                pass
        if iscustomized:
            return
        self.aspect_cluster = aspectcluster

        training_path = os.path.join(base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_train.plk')
        self.load_train_preprocessed_data(training_path)

        stopwords = stop_words()
        chi_cal = CHI([" ".join(s.words) for s in self.train_data],
                      [s.aspect_cluster for s in self.train_data],
                      stop_words())

        chi_dict = {}
        for aspect_cluster, feature_list in chi_cal.chi_dict.items():
            chi_dict[aspect_cluster] = feature_list[0: int(len(feature_list) * ratio)]

        for sample in self.train_data:
            sample.bow_words = []
            sample.bow_tags = []
            for w in sample.words:
                if w in stopwords:
                    continue
                if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                    sample.bow_words.append(w)
                    sample.bow_tags.append(w)

        start_time = time.time()
        test_path = os.path.join(base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_test.plk')
        self.load_test_preprocessed_data(test_path)

        for sample in self.test_data:
            sample.bow_words = []
            sample.bow_tags = []
            for w in sample.words:
                if w in stopwords:
                    continue
                if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                    sample.bow_words.append(w)
                    sample.bow_tags.append(w)
        end_time = time.time()
        self.process_time = end_time - start_time


    def randomsplit(self):
        aspectcluster = self.aspect_cluster
        for i in range(len(self.train_data)):
            self.train_data[i].aspect_cluster = np.random.randint(0, aspectcluster)
        for i in range(len(self.test_data)):
            self.test_data[i].aspect_cluster = np.random.randint(0, aspectcluster)
        # self.save_as_tmp()

    def reSplit(self, testnum=15):

        self.numsvec = self.getNumsVec(self.test_data)
        currentnum = self.aspect_cluster
        for i in range(len(self.numsvec)):
            if self.numsvec[i] >= 2 * testnum:
                renum = self.numsvec[i] // testnum
                subdataset = self.subDataset(i, renum)
                ac, _ = aspect_cluster(subdataset, renum, i, currentnum)
                renum = ac.n_clusters
                print(f"{self.base_dir} aspectid: {i} reSplit from {currentnum} to {currentnum + renum - 2}")
                word_cluster(subdataset, self.aspect_cluster)
                tempnumsvec = self.getNumsVec(self.test_data)

                currentnum += renum - 1
        self.aspect_cluster = currentnum
        self.save_as_pickle()
        self.save_as_txt()
        # self.save_as_tmp()

    def reSplitfromvec(self, vector=[], testnum=10):
        currentnum = self.aspect_cluster
        self.numsvec = self.getNumsVec(self.test_data)
        for i in range(len(vector)):
            aspectid = vector[i]
            renum = math.ceil(self.numsvec[aspectid] / 10)
            subdataset = self.subDataset(aspectid, testnum)
            ac, _ = aspect_cluster(subdataset, renum, aspectid, currentnum)
            renum = ac.n_clusters
            print(f"{self.base_dir} aspectid: {aspectid} reSplit from {currentnum} to {currentnum + renum - 2}")
            word_cluster(subdataset, self.aspect_cluster)
            tempnumsvec = self.getNumsVec(self.test_data)
            currentnum += renum - 1
        self.aspect_cluster = currentnum
        self.save_as_pickle()
        self.save_as_txt()

    def subDataset(self, aspectid, aspectcluster):
        traindata = []
        testdata = []
        for i in range(len(self.train_data)):
            if self.train_data[i].aspect_cluster == aspectid:
                traindata.append(self.train_data[i])
        for i in range(len(self.test_data)):
            if self.test_data[i].aspect_cluster == aspectid:
                testdata.append(self.test_data[i])
        subdata = Dataset(self.base_dir, True, iscustomized=True)
        subdata.aspect_cluster = aspectcluster
        subdata.train_data = traindata
        subdata.test_data = testdata
        return subdata

    def load_raw_data(self, path):
        data = []
        lines = read_as_list(path)
        for i in range(len(lines) // 3):
            data.append(Sample(lines[i * 3], lines[i * 3 + 1], int(lines[i * 3 + 2])))
            if self.aspectword in data[i].text:
                data[i].text = data[i].text.replace(self.aspectword, '##')
        return data

    def getNumsVec(self, data):
        numsvec = [0 for _ in range(self.aspect_cluster)]
        for sap in data:
            if sap.aspect_cluster >= 0 and sap.aspect_cluster < self.aspect_cluster:
                numsvec[sap.aspect_cluster] += 1
        sum = np.sum(numsvec)
        return numsvec

    def load_train_preprocessed_data(self, train_path):
        self.train_data = pickle.load(open(train_path, 'rb'))

    def load_test_preprocessed_data(self, test_path):
        self.test_data = pickle.load(open(test_path, 'rb'))

    def save_as_pickle(self):
        plkdir = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster))
        training_path = os.path.join(plkdir, 'parsed_train.plk')
        test_path = os.path.join(plkdir, 'parsed_test.plk')
        if os.path.isdir(plkdir) is False:
            make_dirs(plkdir)
        pickle.dump(self.train_data, open(training_path, 'wb'))
        pickle.dump(self.test_data, open(test_path, 'wb'))

    def save_as_txt(self):
        training_path = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_train.txt')
        test_path = os.path.join(self.base_dir, 'parsed_data', str(self.aspect_cluster), 'parsed_test.txt')
        with open(training_path, 'w') as f:
            for sample in self.train_data:
                f.write(sample.__str__())

        with open(test_path, 'w') as f:
            for sample in self.test_data:
                f.write(sample.__str__())

    def data_from_aspect(self, aspect_cluster, is_sampling=True):
        pos = 0
        neg = 0
        net = 0
        train_samples = []
        for s in self.train_data:
            if s.aspect_cluster == aspect_cluster:
                if s.polarity == 1:
                    pos += 1
                elif s.polarity == 0:
                    net += 1
                else:
                    neg += 1
                train_samples.append(s)

        if is_sampling:
            # if pos == 0:
            #     maxnum = max(pos, net, neg, 1)
            #     for s in self.train_data:
            #         if s.polarity == 1 and s.aspect_cluster != aspect_cluster:
            #             train_samples.append(s)
            #             pos += 1
            #         if pos >= maxnum:
            #             break
            if net < pos:
                for s in self.train_data:
                    if s.polarity == 0 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        net += 1
                    if net >= pos:
                        break
            if neg < pos:
                for s in self.train_data:
                    if s.polarity == -1 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        neg += 1
                    if neg >= pos:
                        break
        # start_time = time.time()
        test_samples = [s for s in self.test_data if s.aspect_cluster == aspect_cluster]
        # end_time = time.time()

        return train_samples, test_samples

    def get_aspect_labels(self):
        return list(set([s.aspect_cluster for s in self.train_data]))

    def save_as_tmp(self):
        cluster_path = os.path.join(self.base_dir, "aspect_cluster", str(self.aspect_cluster))
        if os.path.isdir(cluster_path) is False:
            make_dirs(cluster_path)
        train_path = os.path.join(cluster_path, 'train')
        test_path = os.path.join(cluster_path, 'test')
        if os.path.isdir(train_path) is False:
            make_dirs(train_path)
        if os.path.isdir(test_path) is False:
            make_dirs(test_path)
        for s in self.train_data:
            with open(os.path.join(train_path, str(s.aspect_cluster)), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")
        for s in self.test_data:
            with open(os.path.join(test_path, str(s.aspect_cluster)), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")


class Sample(object):
    def __init__(self, text, aspect, polarity):
        self.text = text
        self.aspect = aspect
        self.polarity = polarity
        self.words = []
        self.pos_tags = []
        self.dependent_words = []   # words that has dependency with aspect
        self.dependent_pos_tags = []
        self.aspect_cluster = -1
        self.bow_words = []
        self.bow_tags = []
        self.sbow_vec = []
        self.vector = None
    def __str__(self):
        result = "###############################################################\n" + \
                 self.text + '\n' + self.aspect + '\n' + str(self.polarity) + '\n' + \
                 str(self.aspect_cluster) + '\n' + " ".join(self.words) + '\n' + " ".join(self.pos_tags)\
                 + '\n' + " ".join(self.dependent_words) + '\n' + " ".join(self.dependent_pos_tags) + '\n'\
                 "###############################################################\n"

        return result

def calcInstanceNums(index = -1):
    # dataaspect = [49, 20, 30, 30, 40, 20]
    dataaspect = [49, 20, 30, 30, 99, 20]
    base_dirs = ['datasets/laptops/', 'datasets/rest', 'datasets/rest15', 'datasets/rest16', 'datasets/MAMS', 'datasets/twitter']
    for i in range(len(base_dirs)):
        if index >= 0 and i != index:continue
        pos = 0
        neu = 0
        neg = 0
        data = Dataset(base_dirs[i], is_preprocessed=True, aspectcluster=dataaspect[i])
        for s in data.train_data:
            if s.polarity == 1:
                pos += 1
            elif s.polarity == 0:
                neu += 1
            elif s.polarity == -1:
                neg += 1
        print(f"{base_dirs[i]} train: pos {pos} neu {neu} neg {neg}")
        pos = 0
        neu = 0
        neg = 0
        for s in data.test_data:
            if s.polarity == 1:
                pos += 1
            elif s.polarity == 0:
                neu += 1
            elif s.polarity == -1:
                neg += 1
        print(f"{base_dirs[i]} test: pos {pos} neu {neu} neg {neg}")

def splitAll(index = -1):
    base_dirs = ['datasets/laptops/', 'datasets/rest', 'datasets/rest15', 'datasets/rest16', 'datasets/MAMS', 'datasets/twitter']
    splitnum = [15, 30, 15, 15, 30, 15]
    # dataaspect = [30, 20, 30, 30, 40, 20]
    dataaspect = [30, 20, 30, 30, 99, 20]
    for i in range(len(base_dirs)):
        if index >= 0 and i != index:continue
        data = Dataset(base_dirs[i], is_preprocessed=True, aspectcluster=dataaspect[i])
        data.getNumsVec(data.train_data)
        data.getNumsVec(data.test_data)
        data.reSplit(splitnum[i])

def processAll(index = -1, aspectword='##'):
    # dataaspect = [30, 20, 30, 30, 40, 20]
    dataaspect = [30, 20, 30, 30, 99, 20]
    aspectwords = ['$T$', '##', '##', '##', '##', '##' ]
    base_dirs = ['datasets/laptops/', 'datasets/rest', 'datasets/rest15', 'datasets/rest16', 'datasets/MAMS', 'datasets/twitter']
    for i in range(len(base_dirs)):
        if index >= 0 and i != index:continue
        if index == -1 :
            aspectword = aspectwords[i]
        data = Dataset(base_dirs[i], is_preprocessed=False, aspectcluster=dataaspect[i], aspectword=aspectword)
        data.getNumsVec(data.train_data)
        data.getNumsVec(data.test_data)
    
def processMAMSfromvec():
    base_dirs = 'datasets/MAMS'
    aspects = [43, 4, 6, 16, 2, 42, 18, 20, 47, 5, 3, 35, 25, 29]
    data = Dataset(base_dirs, is_preprocessed=True, aspectcluster=48)
    print("data loaded")
    data.reSplitfromvec(aspects)

def process1():
    dataaspect = [1, 1]
    base_dirs = ['datasets/rest'] #'datasets/laptops/', 
    for i in range(len(base_dirs)):
        # if i <= 1:continue
        data = Dataset(base_dirs[i], is_preprocessed=False, aspectcluster=dataaspect[i])
        data.getNumsVec(data.train_data)
        data.getNumsVec(data.test_data)


def randomLaptop14(aspects):
    dataaspect = [aspects]
    base_dirs = ['datasets/laptops/']
    data = Dataset(base_dirs[0], is_preprocessed=False, aspectcluster=dataaspect[0], israndom=True)

if __name__ == '__main__':
    # processMAMSfromvec()
    # process1()
    # randomLaptop14(30)
    # randomLaptop14(46)
    processAll(index=4, aspectword='$T$')
    splitAll(index=4)
        

            
        
