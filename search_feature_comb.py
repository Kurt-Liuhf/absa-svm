from dataset import *
from file_utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from lexicon_features import *
from hyperopt_svm import HyperoptTuner
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale


REST_DIR = 'datasets/rest/'
LAPTOP_DIR = 'datastes/laptops/'
stop_words = stop_words()


def generate_vectors(train_data, test_data, bf, lsa_k=None):
    if bf == 'all_words':
        x_train_tfidf, x_test_tfidf = dependent_features_vectors([s.words for s in train_data],
                                                                 [s.words for s in test_data])
    elif bf == 'parse_result':
        x_train_tfidf, x_test_tfidf = dependent_features_vectors([s.dependent_words for s in train_data],
                                                                 [s.dependent_words for s in test_data])
    elif bf == 'parse+chi':
        x_train_tfidf, x_test_tfidf = bow_features_vectors([s.bow_words for s in train_data],
                                                           [s.bow_words for s in test_data])

    if lsa_k is not None and lsa_k != 'no':
        svd = TruncatedSVD(lsa_k, algorithm='arpack', random_state=42, n_iter=5000)
        lsa = make_pipeline(svd)
        x_train_tfidf = lsa.fit_transform(x_train_tfidf)
        x_test_tfidf = lsa.transform(x_test_tfidf)
        print('!!!!!!!!!!!!!!!!!!!!!????????????????????????????????')

    x_train_sbow = np.asarray([s.sbow_vec for s in train_data])
    x_test_sbow = np.asarray([s.sbow_vec for s in test_data])

    x_train_lfe = lexicons_features_vectors([s.words for s in train_data],
                                            [s.pos_tags for s in train_data],
                                            [s.bow_words for s in train_data])
    x_test_lfe = lexicons_features_vectors([s.words for s in test_data],
                                           [s.pos_tags for s in test_data],
                                           [s.bow_words for s in test_data])

    x_train = np.concatenate((x_train_tfidf, x_train_sbow, x_train_lfe), axis=1)
    x_test = np.concatenate((x_test_tfidf, x_test_sbow, x_test_lfe), axis=1)
    y_train = [y.polarity for y in train_data]
    y_test = [y.polarity for y in test_data]

    return x_train, y_train, x_test, y_test



def dependent_features_vectors(train_words, test_words):
    new_train_texts = []
    new_test_texts = []
    # get chi feature words

    for words in train_words:
        new_words = [w for w in words if w not in stop_words]
        new_train_texts.append(" ".join(new_words))
    for words in test_words:
        new_words = [w for w in words if w not in stop_words]
        new_test_texts.append(" ".join(new_words))
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(new_train_texts).toarray()
    x_test_tfidf = tfidf_vectorize.transform(new_test_texts).toarray()

    return x_train_tfidf, x_test_tfidf


def bow_features_vectors(train_sentences, test_sentences):
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(train_sentences).toarray()
    x_test_tfidf = tfidf_vectorize.transform(test_sentences).toarray()

    return x_train_tfidf, x_test_tfidf


def lexicons_features_vectors(tokens, pos_tags, dependent_words=None):
    new_tokens = []
    new_pos_tags = []
    for words, tags, dw in zip(tokens, pos_tags, dependent_words):
        new_words = []
        new_tags = []
        tmp_dw_set = set(dw.split())
        for w, t in zip(words, tags):
            # if w in tmp_dw_set:
            new_words.append(w)
            new_tags.append(t)
        new_tokens.append(new_words)
        new_pos_tags.append(new_tags)
    return LexiconFeatureExtractor(new_tokens, new_pos_tags).vectors


def evaluation(y_preds, y_true):
    acc = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average='macro')
    clf_report = classification_report(y_true, y_preds)
    print("\n\n################################################################")
    print('Optimized acc: %.5f ' % acc)
    print('Optimized macro_f1: %.5f ' % f1)
    print(clf_report)
    print("####################################################################")


def main():
    chi_ratios = [x/10 for x in range(1, 11)]
    bow_features = ['all_words', 'parse_result', 'parse+chi']  #
    is_sampling = [True, False]
    best_accs = [0 for _ in range(0, 10)]
    print(chi_ratios)
    for aspect_id in range(0, 10):
        ht = HyperoptTuner()
        for cr in chi_ratios:
            data = Dataset(base_dir=REST_DIR, is_preprocessed=True, ratio=cr)
            for iss in is_sampling:
                train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=iss)
                print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                      (aspect_id, len(train_data), len(test_data)))
                for bf in bow_features:
                    x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
                    scaler =Normalizer().fit(x_train)
                    x_train = scaler.transform(x_train)
                    x_test = scaler.transform(x_test)
                    ht.train_X = x_train
                    ht.train_y = y_train
                    ht.test_X = x_test
                    ht.test_y = y_test
                    ht.cluster_id = aspect_id
                    ht.base_dir = data.base_dir
                    ht.tune_params(1000)

                    if ht.best_acc > best_accs[aspect_id]:
                        best_accs[aspect_id] = ht.best_acc
                        with open('svm_' + str(aspect_id), 'w') as f:
                            f.write("################################################################\n")
                            # f.write('chi_ratio: ' + str(cr) + '\n')
                            f.write('cr: ' + str(cr) + '\n')
                            f.write('bow_features: ' + bf + '\n')
                            f.write('is_sampling: ' + str(iss) + '\n')
                            f.write(str(ht.best_cfg) + "\n")
                            f.write('Optimized acc: %.5f \n' % ht.best_acc)
                            f.write('Optimized macro_f1: %.5f \n' % ht.best_f1)
                            f.write('training set shape: %s\n' % str(ht.train_X.shape))
                            f.write(ht.clf_report)
                            f.write("correct / total: %d / %d\n" % (ht.correct, len(ht.test_y)))
                            f.write(str(ht.elapsed_time) + "\n")
                            f.write("################################################################")



# def main():
    # chi_ratios = [x/10 for x in range(1, 11)]
    # bow_features = ['all_words', 'parse_result']  # , 'parse+chi'
    # is_sampling = [True, False]
    # lsa_ks = [10, 15, 20, 50, 100, 200, 300]
    # best_accs = [0 for _ in range(0, 10)]
    # data = Dataset(base_dir=REST_DIR, is_preprocessed=True)
    # for aspect_id in range(0, 10):
        # for lsa_k in lsa_ks:
            # ht = HyperoptTuner()
            # for iss in is_sampling:
                # train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=iss)
                # print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                      # (aspect_id, len(train_data), len(test_data)))
                # for bf in bow_features:
                    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, lsa_k)
                    # scaler = Normalizer().fit(x_train)
                    # x_train = scaler.transform(x_train)
                    # x_test = scaler.transform(x_test)
                    # ht.train_X = x_train
                    # ht.train_y = y_train
                    # ht.test_X = x_test
                    # ht.test_y = y_test
                    # ht.cluster_id = aspect_id
                    # ht.base_dir = data.base_dir
                    # ht.tune_params(1000)

                    # if ht.best_acc > best_accs[aspect_id]:
                        # best_accs[aspect_id] = ht.best_acc
                        # with open('svm_' + str(aspect_id), 'w') as f:
                            # f.write("################################################################\n")
                            # # f.write('chi_ratio: ' + str(cr) + '\n')
                            # f.write('lsa_k: ' + str(lsa_k) + '\n')
                            # f.write('bow_features: ' + bf + '\n')
                            # f.write('is_sampling: ' + str(iss) + '\n')
                            # f.write(str(ht.best_cfg) + "\n")
                            # f.write('Optimized acc: %.5f \n' % ht.best_acc)
                            # f.write('Optimized macro_f1: %.5f \n' % ht.best_f1)
                            # f.write('training set shape: %s\n' % str(ht.train_X.shape))
                            # f.write(ht.clf_report)
                            # f.write("correct / total: %d / %d\n" % (ht.correct, len(ht.test_y)))
                            # f.write(str(ht.elapsed_time) + "\n")
                            # f.write("################################################################")


if __name__ == '__main__':
    main()
