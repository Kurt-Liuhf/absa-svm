from dataset_depparse import *
from file_utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lexicon_features import *
# from hyperopt_svm import HyperoptTuner
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from hyperopt_libsvm import HyperoptTunerLibSVM
from sklearn.svm import SVC as LibSVC
from thundersvm import SVC
import time


REST_DIR = 'datasets/rest/'
LAPTOP_DIR = 'datastes/laptops/'
stop_words = stop_words()


def generate_vectors(train_data, test_data, bf, lsa_k=None):
    if bf == 'all_words':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec = dependent_features_vectors([s.words for s in train_data],
                                                                 [s.words for s in test_data],
                                                                 [s.pos_tags for s in train_data],
                                                                 [s.pos_tags for s in test_data])
    elif bf == 'parse_result':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec  = dependent_features_vectors([s.dependent_words for s in train_data],
                                                                 [s.dependent_words for s in test_data],
                                                                 [s.dependent_pos_tags for s in train_data],
                                                                 [s.dependent_pos_tags for s in test_data])
    elif bf == 'parse+chi':
        # x_train_tfidf, x_test_tfidf, _, _= bow_features_vectors([s.bow_words for s in train_data],
                                                           # [s.bow_words for s in test_data])
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec  = dependent_features_vectors([s.bow_words for s in train_data],
                                                     [s.bow_words for s in test_data],
                                                     [s.bow_tags for s in train_data],
                                                     [s.bow_tags for s in test_data])

    if lsa_k is not None and lsa_k != 'no':
        svd = TruncatedSVD(lsa_k, algorithm='arpack', random_state=42, n_iter=5000)
        lsa = make_pipeline(svd)
        x_train_tfidf = lsa.fit_transform(x_train_tfidf)
        x_test_tfidf = lsa.transform(x_test_tfidf)

    x_train_sbow = np.asarray([s.sbow_vec for s in train_data])
    x_test_sbow = np.asarray([s.sbow_vec for s in test_data])

    x_train_lfe = lexicons_features_vectors([s.words for s in train_data],
                                            [s.pos_tags for s in train_data],
                                            [s.dependent_words for s in train_data])
    x_test_lfe = lexicons_features_vectors([s.words for s in test_data],
                                           [s.pos_tags for s in test_data],
                                           [s.dependent_words for s in test_data])

    x_train = np.concatenate((x_train_tfidf, x_train_pos_vec,  x_train_sbow, x_train_lfe), axis=1)
    x_test = np.concatenate((x_test_tfidf, x_test_pos_vec, x_test_sbow, x_test_lfe), axis=1)
    y_train = [y.polarity for y in train_data]
    y_test = [y.polarity for y in test_data]
    return x_train, y_train, x_test, y_test


def dependent_features_vectors(train_words, test_words, train_pos_tags=None, test_pos_tags=None):
    new_train_texts = []
    new_test_texts = []

    for words in train_words:
        new_words = [w for w in words if w not in stop_words]
        new_train_texts.append(" ".join(new_words))
    for words in test_words:
        new_words = [w for w in words if w not in stop_words]
        new_test_texts.append(" ".join(new_words))
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(new_train_texts).toarray()
    x_test_tfidf = tfidf_vectorize.transform(new_test_texts).toarray()

    # add pos tags information
    x_train_pos_vec = []
    x_test_pos_vec = []
    if train_pos_tags is not None and test_pos_tags is not None:
        count_vectorize = CountVectorizer(token_pattern=r'\w{1,}', binary=False)
        new_train_pos = [" ".join(x) for x in train_pos_tags]
        new_test_pos = [" ".join(x) for x in test_pos_tags]
        x_train_pos_vec = count_vectorize.fit_transform(new_train_pos).toarray()
        x_test_pos_vec = count_vectorize.transform(new_test_pos).toarray()

    return x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec


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
        # tmp_dw_set = set([w for w in dw if w not in stop_words])
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
    total_params = 0
    total_sv = 0

    for aspect_id in range(0, 20):
        path_to_load = f"/data1/DATA/hanfeng/PycharmProjects/svm-result/svm-results-k20/svm_{aspect_id}"
        chi_ratio = 0.0
        bow_feature_type = None
        is_sampling = True

        with open(path_to_load) as f:
            for line in f.readlines():
                line = line.strip()
                if "chi_ratio" in line:
                    chi_ratio = float(line.split(" ")[1])
                elif "bow_features" in line:
                    bow_feature_type = line.split(" ")[1]
                elif "is_sampling" in line:
                    tmp = line.split(" ")[1]
                    is_sampling = True if tmp == "True" else False
                elif "{" in line:
                    import json
                    import ast
                    # kwargs = json.loads(line)
                    kwargs = ast.literal_eval(line)

        print(f"chi_ratio: {chi_ratio}, bow_feature_type: {bow_feature_type}, is_sampling: {is_sampling}, kwargs: {kwargs}")

        if 'chi' in bow_feature_type:
            data = Dataset(base_dir=REST_DIR, is_preprocessed=True, ratio=chi_ratio) #
        else:
            data = Dataset(base_dir=REST_DIR, is_preprocessed=True) #
        train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=is_sampling)
        print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
              (aspect_id, len(train_data), len(test_data)))
        x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bow_feature_type)
        print(x_train.shape)
        scaler = Normalizer().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        clf = SVC(**kwargs)
        clf.fit(x_train, y_train)
        y_preds = clf.predict(x_test)
        evaluation(y_preds, y_test)
        num_sv = clf.support_vectors_.shape[0]
        sv_dims = clf.support_vectors_.shape[1]
        print(f"size of support vector: ({num_sv}, {sv_dims})")
        total_params += num_sv * sv_dims
        total_sv += num_sv

    print(f"total params: {total_params}")
    print(f"total support_vectors: {total_sv}")


if __name__ == '__main__':
    main()
