from collections import Counter
from dataset_depparse import *
from file_utils import *


class CHI(object):

    def __init__(self, texts, labels, stop_words=[]):
        self.texts = texts
        self.labels = labels
        self.n_instances = len(texts)
        self.labels_counter = self._labels_count()
        self.stop_words = set(stop_words)
        self.chi_dict = self.calculate_chi()

    def _labels_count(self):
        labels_counter = Counter(self.labels)
        return labels_counter

    def _label_word_count(self):
        label_word_counter = {}
        all_words = []
        # set default values of the dictionary
        for label in self.labels_counter.keys():
            label_word_counter[label] = []
        # start counting
        for text, label in zip(self.texts, self.labels):
            words = [x for x in text.split(' ') if x not in self.stop_words]
            label_word_counter[label].extend(set(words))
            all_words.extend(set(words))
        for label in label_word_counter.keys():
            label_word_counter[label] = Counter(label_word_counter[label])
        word_counter = Counter(all_words)

        return label_word_counter, word_counter

    def _score(self, word, label, label_word_counter, word_counter):
        A = label_word_counter[label].get(word, 0)
        B = word_counter.get(word, 0) - A
        C = self.labels_counter[label] - A
        D = self.n_instances - self.labels_counter[label] - B
        score = 0
        if (A + B) * (C + D) != 0:
            score = ((A * D - B * C) * (A * D - B * C)) / ((A + B) * (C + D))

        return score

    def calculate_chi(self):
        ''' returns the chi calculations for each word in the data; importance of each word in the svm relative to its importance in every other svm.
            chi_dict = {
                cluster_id: [(word, chi-score)] // currently ascending order
            }
        '''
        print("begin calculating chi ...")
        chi_dict = {}  # key-label: value-dict
        label_word_counter, word_counter = self._label_word_count()
        # init dict
        for label in self.labels_counter.keys():
            chi_dict[label] = {}

        for word in word_counter.keys():
            for label in self.labels_counter.keys():
                score = self._score(word, label, label_word_counter, word_counter)
                chi_dict[label][word] = score
        print('chi calculation done ...')
        for label in self.labels_counter.keys():
            tmp_dict = chi_dict[label]
            chi_dict[label] = sorted(tmp_dict.items(), key=lambda x: x[1]) # , reverse=True
            # print(chi_dict[label])
        return chi_dict


if __name__ == '__main__':
    data = Dataset(base_dir='datasets/rest', is_preprocessed=True)
    chi = CHI([" ".join(s.words) for s in data.train_data],
              [s.aspect_cluster for s in data.train_data],
              stop_words())
    # print(chi.chi_dict[0])
    # print(len(chi.chi_dict[0]))
