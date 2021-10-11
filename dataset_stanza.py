import stanza
import os
from file_utils import *
import pickle
from cluster_utils import *
from chi import CHI
from typing import List, Tuple

import faulthandler

def aspect_cluster(dataset, n_clusters=20):
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
    def __init__(self, base_dir, is_preprocessed, ratio=0.3):
        self.base_dir = base_dir
        self.nlp_helper = self.load_stanza()
        if not is_preprocessed:
            training_path = os.path.join(base_dir, 'train.txt')
            test_path = os.path.join(base_dir, 'test.txt')
            self.train_data = self.load_raw_data(training_path)
            self.test_data = self.load_raw_data(test_path)

            print(self.train_data[0].__str__())
            self.preprocessing(self.train_data)
            self.preprocessing(self.test_data)

            print('attempt aspect cluster')
            faulthandler.enable()
            aspect_cluster(self)
            print('attempt word cluster')
            word_cluster(self)

            print('save files...')
            self.save_as_pickle(base_dir, 'parsed_data', '_parsed_train.plk', '_parsed_test.plk', self.train_data, self.test_data)
            self.save_as_txt(base_dir, 'parsed_data', '_parsed_train.txt', '_parsed_test.txt', self.train_data, self.test_data)
        else:
            pass

    def load_stanza(self):
        stanza.download('en')
        return stanza.Pipeline(lang='en', tokenize_pretokenized=True)

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
    
    def preprocessing(self, data):
        length = len(data)
        for i, sample in enumerate(data):
            print(sample.text)
            print(sample.aspect)
            # 1. tokenize && pos tagging
            # sample.words, sample.pos_tags = nlp_helper.pos_tag(sample.text)
            nlp_parsed_obj = self.nlp_helper(sample.text)
            sample.words, sample.pos_tags = list(map(list, zip(
                *[(word.text, word.xpos) for sent in nlp_parsed_obj.sentences for word in sent.words])))
            # 2. get aspect-dependent words

            aspect_term = sample.aspect.split(' ')[-1]
            sample.words, idx, aspect_term = self.format_hashstring(sample.words, aspect_term)
            sample.aspect = aspect_term
            tmp_text = str.replace(sample.text, '##', aspect_term)
            dependencies = [(dep_edge[1], dep_edge[0].id, dep_edge[2].id)
                            for sent in nlp_parsed_obj.sentences for dep_edge in sent.dependencies]
            sample.dependent_words, sample.dependent_pos_tags, _ = self.get_dependent_words(
                idx, dependencies, sample.words, sample.pos_tags, tmp_text, n=3, window_size=5)
            #print(sample)
            
            print(f'progress: {round(((i+1) / length * 100), 2)}% --- {i}')

            break
            #break

    def direction_dependent(self, temp_dict, word, n):
        selected_words = []
        if word not in temp_dict.keys():
            return []
        else:
            tmp_list = temp_dict[word]
            selected_words.extend(tmp_list)
            if n > 1:
                for w in tmp_list:
                    selected_words.extend(self.direction_dependent(temp_dict, w, n - 1))
        return selected_words

    def format_hashstring(self, tokens: List[str], aspect: str) -> Tuple[List[str], int, str]:
        # fix malformed hashwords, e.g. '*##', '##*', '*##*'
        try:
            idx = [i for i, tokens in enumerate(tokens) if '##' in tokens][0]
            hashword = tokens[idx]
            tokens[idx] = '##'
        except IndexError as e:
            print(e)
            idx = None
            hashword = None
        aspect_trunc = ''
        if len(hashword) > 2:
            if hashword.startswith('#'): # "##*"
                aspect_trunc = hashword.split('##')[1]
                aspect = aspect + aspect_trunc
            elif hashword.endswith('#'): # "*##"
                aspect_trunc = hashword.split('##')[0] 
                aspect = aspect_trunc + aspect
            else: # "*##*"
                aspect_trunc = hashword.split('##') 
                aspect = aspect.join(aspect_trunc) 
        return tokens, idx, aspect

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

    def get_dependent_words(self, idx, dependent_results, words, pos_tags, text, n=2, window_size=0):
        
        #dependent_results = dependent_parse(text)
        in_dict = {}
        out_dict = {}
        for dr in dependent_results:
            # print(dr[0])
            src_wid = dr[1]    # source wid
            tag_wid = dr[2]    # target wid
            out_dict.setdefault(src_wid, [])
            in_dict.setdefault(tag_wid, [])

            out_dict[src_wid].append(tag_wid)
            in_dict[tag_wid].append(src_wid)

        forwards = self.direction_dependent(out_dict, idx + 1, n)
        backwards = self.direction_dependent(in_dict, idx + 1, n)

        result = []
        result.extend(forwards)
        result.extend(backwards)

        # add window-size words
        if window_size != 0:
            # right side
            for i in range(idx + 2, idx + 2 + window_size, 1):
                if i > len(words):
                    break
                result.append(i)
            for i in range(idx + 1 - window_size, idx + 1, 1):
                if i > 1:
                    result.append(i)
        result = list(set(result))
        result.sort()

        #print("!!!!!!!--->> " + " ".join(pos_tags))
        print("!!!!!!!--->> " + " ".join([pos_tags[i-1] for i in result]))

        return [words[i-1] for i in result], [pos_tags[i-1] for i in result], dependent_results


    def save_as_pickle(self, base_dir, fp, fname_tr, fname_t, train_data, test_data):
        training_path = os.path.join(base_dir, fp, fname_tr)
        test_path = os.path.join(base_dir, fp, fname_t)
        pickle.dump(train_data, open(training_path, 'wb'))
        pickle.dump(test_data, open(test_path, 'wb'))

    def save_as_txt(self, base_dir, fp, fname_tr, fname_t, train_data, test_data):
        training_path = os.path.join(base_dir, fp, fname_tr)
        test_path = os.path.join(base_dir, fp, fname_t)
        with open(training_path, 'w') as f:
            for sample in train_data:
                f.write(sample.__str__())
        with open(test_path, 'w') as f:
            for sample in test_data:
                f.write(sample.__str__())

    def save_as_tmp(self, base_dir, fp_tr, fp_t, train_data, test_data):
        remove_dirs(base_dir)
        make_dirs(os.path.join(base_dir, 'train'))
        make_dirs(os.path.join(base_dir, 'test'))
        for s in train_data:
            with open(f"{base_dir}/{fp_tr}/" + str(s.aspect_cluster), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")
        for s in test_data:
            with open(f"{base_dir}/{fp_t}/" + str(s.aspect_cluster), 'a') as f:
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

    def __str__(self):
        result = "###############################################################\n" + \
                 self.text + '\n' + self.aspect + '\n' + str(self.polarity) + '\n' + \
                 str(self.aspect_cluster) + '\n' + " ".join(self.words) + '\n' + " ".join(self.pos_tags)\
                 + '\n' + " ".join(self.dependent_words) + '\n' + " ".join(self.dependent_pos_tags) + '\n'\
                 "###############################################################\n"

        return result



# if __name__ == '__main__':
#     base_dir = 'datasets/rest/'
#     data = Dataset(base_dir, is_preprocessed=False)


if __name__ == "__main__":
    base_dir = 'datasets/rest/'
    data = Dataset(base_dir, is_preprocessed=False)