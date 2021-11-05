from dataset_stanza import *
from file_utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from lexicon_features import *
# from hyperopt_svm import HyperoptTuner
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from hyperopt_libsvm import HyperoptTunerLibSVM


REST_DIR = 'datasets/rest/'
LAPTOP_DIR = 'datastes/laptops/'
stop_words = stop_words()


def generate_vectors(train_data, test_data):
    x_train_tfidf, x_test_tfidf = dependent_features_vectors([s.words for s in train_data],
                                                             [s.words for s in test_data])
    # x_train_tfidf, x_test_tfidf = bow_features_vectors([s.bow_words for s in train_data],
    #                                                    [s.bow_words for s in test_data])

    x_train_sbow = np.asarray([s.sbow_vec for s in train_data])
    x_test_sbow = np.asarray([s.sbow_vec for s in test_data])

    x_train_lfe = lexicons_features_vectors([s.words for s in train_data],
                                            [s.pos_tags for s in train_data],
                                            [s.bow_words for s in train_data])
    x_test_lfe = lexicons_features_vectors([s.words for s in test_data],
                                           [s.pos_tags for s in test_data],
                                           [s.bow_words for s in test_data])

    x_train = np.concatenate((x_train_tfidf, x_train_lfe), axis=1)
    x_test = np.concatenate((x_test_tfidf, x_test_lfe), axis=1)
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
        #tmp_dw_set = set(dw.split())
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
    data = Dataset(base_dir=REST_DIR, is_preprocessed=True)
    aspect_labels = data.get_aspect_labels()
    print(aspect_labels)
    y_preds = []
    y_true = []
    for aspect_id in range(0, 20):  # NOTE: CHANGE THIS RANGE TO REFLECT NUMBER OF CLUSTERS
        train_data, test_data = data.data_from_aspect(aspect_id)
        print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
              (aspect_id, len(train_data), len(test_data)))
        x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data)
        scaler = Normalizer().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        # ht = HyperoptTuner(x_train, y_train, x_test, y_test, aspect_id, data.base_dir)
        ht = HyperoptTunerLibSVM(x_train, y_train, x_test, y_test, aspect_id, data.base_dir)
        ht.tune_params(500)
        y_preds.extend(ht.pred_results)
        y_true.extend(y_test)
    evaluation(y_preds, y_true)


if __name__ == '__main__':
    main()
