from file_utils import *
from stanford_nlp import StanfordNLP
import pickle
from cluster_utils import *
from chi import CHI


def preprocessing(data):
    nlp_helper = StanfordNLP()

    for sample in data:
        # 1. tokenize && pos tagging
        sample.words, sample.pos_tags = nlp_helper.pos_tag(sample.text)
        # 2. get aspect-dependent words
        aspect_term = sample.aspect.split(' ')[-1]
        tmp_text = str.replace(sample.text, '##', aspect_term)
        sample.dependent_words, _ = nlp_helper.get_dependent_words(sample.words, tmp_text, n=2, window_size=5)
        print(sample)


def aspect_cluster(dataset, n_clusters=10):
    ac = AspectCluster(dataset, n_clusters)
    _, vectors = ac.fit()
    ac.predict()
    ac.save_cluster_result()

    return ac, vectors


def word_cluster(dataset):
    wc = WordsCluster(dataset)
    wc.generate_vector()


def chi_calculation(dataset, ratio):
    stopwords = stop_words()
    chi_cal = CHI([" ".join(s.words) for s in dataset.train_data],
              [s.aspect_cluster for s in dataset.train_data],
              stop_words())

    chi_dict = {}
    for aspect_cluster, feature_list in chi_cal.chi_dict.items():
        chi_dict[aspect_cluster] = feature_list[0: int(len(feature_list) * ratio)]

    for sample in dataset.train_data:
        tmp_words = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                tmp_words.append(w)
        sample.bow_words = " ".join(tmp_words)

    for sample in dataset.test_data:
        tmp_words = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                tmp_words.append(w)
        sample.bow_words = " ".join(set(tmp_words))


class Dataset(object):
    def __init__(self, base_dir, is_preprocessed, ratio=0.3):
        self.base_dir = base_dir
        if not is_preprocessed:
            training_path = os.path.join(base_dir, 'train.txt')
            test_path = os.path.join(base_dir, 'test.txt')
            self.train_data = self.load_raw_data(training_path)
            self.test_data = self.load_raw_data(test_path)
            preprocessing(self.train_data)
            preprocessing(self.test_data)
            aspect_cluster(self)
            word_cluster(self)
            self.save_as_pickle()
            self.save_as_txt()
        else:
            training_path = os.path.join(base_dir, 'parsed_data', 'parsed_train.plk')
            test_path = os.path.join(base_dir, 'parsed_data', 'parsed_test.plk')
            self.load_preprocessed_data(training_path, test_path)
            chi_calculation(self, ratio)

    @staticmethod
    def load_raw_data(path):
        data = []
        lines = read_as_list(path)
        for i in range(len(lines) // 3):
            data.append(Sample(lines[i * 3], lines[i * 3 + 1], int(lines[i * 3 + 2])))

        return data

    def load_preprocessed_data(self, training_path, test_path):
        self.train_data = pickle.load(open(training_path, 'rb'))
        self.test_data = pickle.load(open(test_path, 'rb'))

    def save_as_pickle(self):
        training_path = os.path.join(base_dir, 'parsed_data', 'parsed_train.plk')
        test_path = os.path.join(base_dir, 'parsed_data', 'parsed_test.plk')
        pickle.dump(self.train_data, open(training_path, 'wb'))
        pickle.dump(self.test_data, open(test_path, 'wb'))

    def save_as_txt(self):
        training_path = os.path.join(base_dir, 'parsed_data', 'parsed_train.txt')
        test_path = os.path.join(base_dir, 'parsed_data', 'parsed_test.txt')
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
        test_samples = [s for s in self.test_data if s.aspect_cluster == aspect_cluster]

        return train_samples, test_samples

    def get_aspect_labels(self):
        return list(set([s.aspect_cluster for s in self.train_data]))


class Sample(object):
    def __init__(self, text, aspect, polarity):
        self.text = text
        self.aspect = aspect
        self.polarity = polarity
        self.words = []
        self.pos_tags = []
        self.dependent_words = []   # words that has dependency with aspect
        self.aspect_cluster = -1
        self.bow_words = []
        self.sbow_vec = []

    def __str__(self):
        result = "###############################################################\n" + \
                 self.text + '\n' + self.aspect + '\n' + str(self.polarity) + '\n' + \
                 str(self.aspect_cluster) + '\n' + " ".join(self.words) + '\n' + " ".join(self.pos_tags)\
                 + '\n' + " ".join(self.dependent_words) + '\n' + \
                 "###############################################################\n"

        return result


if __name__ == '__main__':
    base_dir = 'datasets/rest/'
    data = Dataset(base_dir, is_preprocessed=False)
