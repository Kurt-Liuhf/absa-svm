import copy
import getopt
from multiprocessing import Lock
import sys
from FC import featureConstruction
from RL import RLOptimization
from dataset import *
from file_utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lexicon_features import *
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from hyperopt_libsvm import HyperoptTunerLibSVM
from hyperopt_svm import HyperoptTuner
from hyperoptall import HyperoptTunerAll
import time
from MAFC import Dataset as MafcData
from sklearn.svm import SVC
from thundersvm import SVC as TSVC
from FC import featureConstruction
from sklearn.metrics import normalized_mutual_info_score
import warnings
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
REST_DIR = 'datasets/rest/'
LAPTOP_DIR = 'datasets/laptops/'
TWITTER_DIR = 'datasets/twitter/'
REST15_DIR = 'datasets/rest15/'
REST16_DIR = 'datasets/rest16/'
MAMS_DIR = 'datasets/MAMS/'
BETTER = 1.0
LESS = 0.1
stop_words = stop_words()


def generate_vectors(train_data, test_data, bf, lsa_k=None, isabandon=False):
    if bf == 'all_words':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec, test_process_time = dependent_features_vectors([s.words for s in train_data],
                                                                 [s.words for s in test_data],
                                                                 [s.pos_tags for s in train_data],
                                                                 [s.pos_tags for s in test_data])
    elif bf == 'parse_result':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec, test_process_time  = dependent_features_vectors([s.dependent_words for s in train_data],
                                                                 [s.dependent_words for s in test_data],
                                                                 [s.dependent_pos_tags for s in train_data],
                                                                 [s.dependent_pos_tags for s in test_data])
    elif bf == 'parse+chi':
        # x_train_tfidf, x_test_tfidf, _, _= bow_features_vectors([s.bow_words for s in train_data],
                                                           # [s.bow_words for s in test_data])
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec, test_process_time  = dependent_features_vectors([s.bow_words for s in train_data],
                                                     [s.bow_words for s in test_data],
                                                     [s.bow_tags for s in train_data],
                                                     [s.bow_tags for s in test_data])

    if lsa_k is not None and lsa_k != 'no':
        svd = TruncatedSVD(lsa_k, algorithm='arpack', random_state=42, n_iter=5000)
        lsa = make_pipeline(svd)
        x_train_tfidf = lsa.fit_transform(x_train_tfidf)
        start_time = time.time()
        x_test_tfidf = lsa.transform(x_test_tfidf)
        end_time = time.time()
        test_process_time += end_time - start_time


    x_train_sbow = np.asarray([s.sbow_vec for s in train_data])
    start_time = time.time()
    x_test_sbow = np.asarray([s.sbow_vec for s in test_data])
    end_time = time.time()
    test_process_time += end_time - start_time

    x_train_lfe = lexicons_features_vectors([s.words for s in train_data],
                                            [s.pos_tags for s in train_data],
                                            [s.dependent_words for s in train_data])
    start_time = time.time()
    x_test_lfe = lexicons_features_vectors([s.words for s in test_data],
                                           [s.pos_tags for s in test_data],
                                           [s.dependent_words for s in test_data])
    end_time = time.time()
    test_process_time += end_time - start_time
    
    if isabandon:
        x_train = np.concatenate((x_train_tfidf, x_train_pos_vec, x_train_lfe), axis=1)
        start_time = time.time()
        x_test = np.concatenate((x_test_tfidf, x_test_pos_vec, x_test_lfe), axis=1)
        end_time = time.time()
        test_process_time += end_time - start_time
    else:
        x_train = np.concatenate((x_train_tfidf, x_train_pos_vec,  x_train_sbow, x_train_lfe), axis=1)
        start_time = time.time()
        x_test = np.concatenate((x_test_tfidf, x_test_pos_vec, x_test_sbow, x_test_lfe), axis=1)
        end_time = time.time()
        test_process_time += end_time - start_time
    
    y_train = [y.polarity for y in train_data]
    start_time = time.time()
    y_test = [y.polarity for y in test_data]
    end_time = time.time()
    test_process_time += end_time - start_time
    # return x_train, y_train, x_test, y_test, test_process_time
    return x_train, y_train, x_test, y_test



def dependent_features_vectors(train_words, test_words, train_pos_tags=None, test_pos_tags=None):
    test_process_time = 0
    new_train_texts = []
    new_test_texts = []

    for words in train_words:
        new_words = [w for w in words if w not in stop_words]
        new_train_texts.append(" ".join(new_words))

    start_time = time.time()
    for words in test_words:
        new_words = [w for w in words if w not in stop_words]
        new_test_texts.append(" ".join(new_words))
    end_time = time.time()
    test_process_time += end_time - start_time
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(new_train_texts).toarray()
    start_time = time.time()
    x_test_tfidf = tfidf_vectorize.transform(new_test_texts).toarray()
    end_time = time.time()
    test_process_time += end_time - start_time

    # add pos tags information
    x_train_pos_vec = []
    x_test_pos_vec = []
    if train_pos_tags is not None and test_pos_tags is not None:
        count_vectorize = CountVectorizer(token_pattern=r'\w{1,}', binary=False)
        new_train_pos = [" ".join(x) for x in train_pos_tags]
        start_time = time.time()
        new_test_pos = [" ".join(x) for x in test_pos_tags]
        end_time = time.time()
        test_process_time += end_time - start_time
        x_train_pos_vec = count_vectorize.fit_transform(new_train_pos).toarray()
        start_time = time.time()
        x_test_pos_vec = count_vectorize.transform(new_test_pos).toarray()
        end_time = time.time()
        test_process_time += end_time - start_time

    return x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec, test_process_time



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
    return acc, f1

def transFeature(traindata, solution):
    newdata =  copy.deepcopy(traindata)
    if len(traindata) < 1:
        return newdata
    maxj = min(len(solution), len(traindata[0]))
    for i in range(len(traindata)):
        for j in range(maxj):
            newdata[i,j] *= solution[j]
    return newdata

def search(best_f1, aspect_id, bf, rlo, num_rounds, ht, num_rounds2,\
    best_accs, result_path, cr=1, issample=True, ispreprocessed=True,\
        aspect_cluster=20, opttype=2, isfc=False, igrate=0.1,  maxelapsetime=1,\
            isabandon=False, isrl=False, ismatlibplot=False):
    data = Dataset(base_dir=datasetname, is_preprocessed=ispreprocessed, ratio=cr, aspectcluster=aspect_cluster) #
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=issample)
    if len(train_data) == 0 or len(test_data) == 0:
        return
    x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, isabandon=isabandon)
    print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                (aspect_id, len(train_data), len(test_data)))
    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
    print(x_train.shape)
    scaler = Normalizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    fctraindata = x_train
    fctestdata = x_test
    if isfc:
        mafcdata = MafcData(name = datasetname) 
        mafcdata.fromArray(x_train)
        fctraindata = featureConstruction(mafcdata)
        
        mafcdata.fromArray(x_test)
        fctestdata = featureConstruction(mafcdata)
        print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
    fctraindataarray = np.array(fctraindata)
    fctestdataarray = np.array(fctestdata)
    fctraindataarray = np.nan_to_num(fctraindataarray)
    fctestdataarray = np.nan_to_num(fctestdataarray)
    
    scaler2 = Normalizer().fit(fctraindataarray)
    fctraindataarray = scaler2.transform(fctraindataarray)
    fctestdataarray = scaler2.transform(fctestdataarray)
    selectlist = []
    for i in range(len(fctraindataarray[0])):
        if normalized_mutual_info_score(fctraindataarray[:,i], y_train) > igrate:
            selectlist.append(i)
    fctraindataarray = fctraindataarray[:,selectlist]
    fctestdataarray = fctestdataarray[:,selectlist]
    if len(fctraindataarray[0]) == 0 or len(fctestdataarray[0]) == 0:
        return
    model = ht
    if isrl:
        rlo.svmname = ht.modeltype
        model = rlo
    model.train_X = fctraindataarray
    model.train_y = y_train 
    model.test_X = fctestdataarray
    model.test_y = y_test
    model.cluster_id = aspect_id
    model.base_dir = data.base_dir
    model.tune_params(num_rounds2, opttype, maxtimehours=maxelapsetime)
    if model.best_acc >= best_accs[aspect_id]:
        if model.best_acc > best_accs[aspect_id] or model.best_f1 > best_f1[aspect_id]:
            best_accs[aspect_id] = model.best_acc
            best_f1[aspect_id] = model.best_f1
            with open(result_path + 'svm_' + str(aspect_id), 'w') as f:
                f.write("################################################################\n")
                f.write("filter: " + str(igrate) + "\n")
                f.write('model_type: ' + str(ht.modeltype) + '\n')
                f.write('is_fc: ' + str(isfc) + '\n')
                f.write('is_fc2: ' + str(isfc) + '\n')
                f.write('chi_ratio: ' + str(cr) + '\n')
                f.write('cr: ' + str(cr) + '\n')
                f.write('bow_features: ' + bf + '\n')
                f.write('is_sampling: ' + str(issample) + '\n')
                f.write(str(model.best_cfg) + "\n")
                f.write('Optimized acc: %.5f \n' % model.best_acc)
                f.write('Optimized macro_f1: %.5f \n' % model.best_f1)
                f.write('training set shape: %s\n' % str(model.train_X.shape))
                f.write(model.clf_report)
                f.write(str(confusion_matrix(model.test_y, model.best_predict_label, labels=[-1,0,1]))+"\n")
                f.write("correct / total: %d / %d\n" % (model.correct, len(model.test_y)))
                if not isrl and ismatlibplot:
                    x = [i for i in range(len(model.train_error))]
                    plt.plot(x, model.train_error, color='red', linewidth=2.0, linestyle='--')
                    plt.plot(x, model.test_error, color='blue', linewidth=2.0, linestyle='-.')
                    plt.show()
                    plt.savefig(f"{result_path}fig{str(aspect_id)}.png")
                    f.write(f"train_error: {model.train_error}\n")
                    f.write(f"test_error: {model.test_error}\n")
                    
                if isrl:
                    f.write(f"solution: {rlo.solution}\n")
                    f.write(f"lens(sol): {len(rlo.solution)}\n")
                f.write("elapsed time: %.5f s\n" % model.elapsed_time) 
                

def getParams(datasetname, aspect_id, aspect_cluster, modeltype="", othermodel=False):
    path_to_load = root_path + f"{datasetname}/optimal_results/best{aspect_cluster}/svm_{aspect_id}"
    if othermodel:
        path_to_load = root_path + f"{datasetname}/optimal_results/{modeltype}best{aspect_cluster}/svm_{aspect_id}"
    if not os.path.isfile(path_to_load): return {} 
    params = {}
    params['cr'] = 0.3
    params['bf'] = "all_words"
    params['is_sample'] = False
    params['isfc'] = False
    params['isfc2'] = False
    params['model_type'] = "thundersvm"
    params['acc'] = 0
    params['f1'] = 0
    params['solution'] = []
    params['filter'] = -1
    params['elapsedtime'] = 0
    with open(path_to_load) as f:
        for line in f.readlines():
            line = line.strip()
            if "chi_ratio" in line:
                params['cr'] = float(line.split(" ")[1])
            elif "bow_features" in line:
                params['bf'] = line.split(" ")[1]
            elif "is_sampling" in line:
                tmp = line.split(" ")[1]
                params['is_sample'] = True if tmp == "True" else False
            elif "is_fc" in line:
                if "is_fc2" in line:
                    tmp = line.split(" ")[1]
                    params['isfc2'] = True if tmp == "True" else False
                else:
                    tmp = line.split(" ")[1]
                    params['isfc'] = True if tmp == "True" else False
            elif "model_type" in line or "svm_type" in line:
                params['model_type'] = line.split(":")[1]
            elif "Optimized acc" in line:
                params['acc'] = float(line.split(" ")[2])
            elif "Optimized macro_f1" in line:
                params['f1'] = float(line.split(" ")[2])
            elif "solution" in line:
                params['solution'] = ast.literal_eval(line.split(": ")[1])
            elif "filter" in line:
                params['filter'] = float(line.split(" ")[1])
            elif "elapsed time" in line:
                params['elapsedtime'] = float(line.split(" ")[2])
            elif "{" in line:
                import json
                import ast
                # kwargs = json.loads(line)
                params['params'] = ast.literal_eval(line)
    return params

def runRL(datasetname, aspect_id, aspect_cluster, result_path, gpuid, maxiter=500, modeltype="", isabandon=False):
    if "now2v" not in modeltype and ("LibSVM" in modeltype or "svm" in modeltype):
        params = getParams(datasetname, aspect_id, aspect_cluster)
    else:
        params = getParams(datasetname, aspect_id, aspect_cluster, modeltype=modeltype, othermodel=True)
    if len(params) == 0 :return
    cr = params['cr']
    issample = params['is_sample']
    bf = params['bf']
    isfc = params['isfc']
    isfc2 = params['isfc2']
    modeltype = params['model_type']
    svmparams = params['params']
    filterscore = params['filter']
    data = Dataset(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster) #
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=issample)
    if len(train_data) == 0 or len(test_data) == 0:
        return
    x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, isabandon=isabandon)
    print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                (aspect_id, len(train_data), len(test_data)))
    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
    print(x_train.shape)
    scaler = Normalizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    fctraindata = x_train
    fctestdata = x_test
    if isfc:
        mafcdata = MafcData(name = datasetname) 
        mafcdata.fromArray(x_train)
        fctraindata = featureConstruction(mafcdata, isother = isfc2)
        
        mafcdata.fromArray(x_test)
        fctestdata = featureConstruction(mafcdata, isother = isfc2)
        print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
    fctraindataarray = np.array(fctraindata)
    fctestdataarray = np.array(fctestdata)
    # listsolution = [1.0 for _ in range(fctraindataarray.shape[1])]
    if isfc or isfc2:
        fctraindataarray = np.nan_to_num(fctraindataarray)
        fctestdataarray = np.nan_to_num(fctestdataarray)
        scaler2 = Normalizer().fit(fctraindataarray)
        fctraindataarray = scaler2.transform(fctraindataarray)
        fctestdataarray = scaler2.transform(fctestdataarray)
    selectlist = []
    for i in range(len(fctraindataarray[0])):
        if normalized_mutual_info_score(fctraindataarray[:,i], y_train) > filterscore:
            selectlist.append(i)
    fctraindataarray = fctraindataarray[:,selectlist]
    fctestdataarray = fctestdataarray[:,selectlist]
    
    rlo = RLOptimization(train_X=fctraindataarray, train_y=y_train, test_X=fctestdataarray, \
        test_y=y_test, dataname=datasetname, resultpath=result_path, params=params, aspect_id=aspect_id, \
            gpuid=gpuid, maxiter=maxiter)
    rlo.run()
    with open(result_path + 'svm_' + str(aspect_id), 'a') as f:
        f.write("elapsed time: %.5f s\n" % rlo.elapsed_time) 


def runVa(datasetname, aspect_id, aspect_cluster, maxiter=500, modeltype="", isabandon=False, isfc=False, isfc2=False, isother=False, svandtraininstance=[0, 0],gpuid=0):
    if "now2v" not in modeltype and ("LibSVM" in modeltype or "svm" in modeltype):
        params = getParams(datasetname, aspect_id, aspect_cluster)
    else:
        params = getParams(datasetname, aspect_id, aspect_cluster, modeltype=modeltype, othermodel=True)
    if len(params) == 0 :return
    cr = params['cr']
    issample = params['is_sample']
    bf = params['bf']
    isfc = params['isfc']
    isfc2 = params['isfc2']
    modeltype = params['model_type']
    svmparams = params['params']
    acc = params['acc']
    f1 = params['f1']
    solution = params['solution']
    filterscore = params['filter']

    data = Dataset(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster) #
    data1 = DatasetV2(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster)
    process_time1 = data1.process_time
    train_data, test_data, process_time2 = data.data_from_aspect(aspect_id, is_sampling=issample)
    if len(train_data) == 0 or len(test_data) == 0:
        return
    x_train, y_train, x_test, y_test, process_time3 = generate_vectors(train_data, test_data, bf, isabandon=isabandon)
    print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                (aspect_id, len(train_data), len(test_data)))
    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
    print(x_train.shape)
    scaler = Normalizer().fit(x_train)
    x_train = scaler.transform(x_train)
    start_time = time.time()
    x_test = scaler.transform(x_test)
    end_time = time.time()
    process_time4 = end_time - start_time
    fctraindata = x_train
    fctestdata = x_test
    if isfc:
        mafcdata = MafcData(name = datasetname) 
        mafcdata.fromArray(x_train)
        fctraindata = featureConstruction(mafcdata, isother = isfc2)
        mafcdata.fromArray(x_test)
        fctestdata = featureConstruction(mafcdata, isother = isfc2)
        print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
    fctraindataarray = np.array(fctraindata)
    fctestdataarray = np.array(fctestdata)
    # listsolution = [1.0 for _ in range(fctraindataarray.shape[1])]
    if isfc or isfc2:
        fctraindataarray = np.nan_to_num(fctraindataarray)
        fctestdataarray = np.nan_to_num(fctestdataarray)
        scaler2 = Normalizer().fit(fctraindataarray)
        fctraindataarray = scaler2.transform(fctraindataarray)
        fctestdataarray = scaler2.transform(fctestdataarray)
    selectlist = []
    for i in range(len(fctraindataarray[0])):
        if normalized_mutual_info_score(fctraindataarray[:,i], y_train) > filterscore:
            selectlist.append(i)
    fctraindataarray = fctraindataarray[:,selectlist]
    start_time = time.time()
    fctestdataarray = fctestdataarray[:,selectlist]
    end_time = time.time()
    process_time5 = end_time - start_time
    
    clf = None
    if "LibSVM" in modeltype:
        clf =  SVC(**svmparams, random_state=42, max_iter=maxiter)
    else:
        clf =  TSVC(**svmparams, random_state=42, max_iter=maxiter, n_jobs=8, gpu_id=gpuid)
    X = fctraindataarray
    Y = y_train
    X_test = fctestdataarray
    if len(solution) > 0:
        selectlist = [i for i, n in enumerate(solution) if n == 1]
        X = X[:,selectlist]
        X_test = X_test[:,selectlist]
    
    print(f"train shape: {X.shape} test shape: {X_test.shape}\n")
    clf.fit(X, Y)
    if "LibSVM" in modeltype:
        print(f"num support_vectors_: {clf.n_support_}")
        svandtraininstance[0] += np.sum(clf.n_support_)
    else:
        print(f"num support_vectors_: {clf.n_sv}")
        svandtraininstance[0] += clf.n_sv
    svandtraininstance[1] += X.shape[0]
    start_time = time.time()
    pred = clf.predict(X_test)
    vaacc, vaf1 = evaluation(pred, y_test)
    if vaacc == acc and vaf1 == f1:
        print(f"aspect_id: {aspect_id} validate suceessful!\n")
    else:
        print(f"aspect_id: {aspect_id} validate failed!, acc:{acc} vs vaacc:{vaacc}; f1:{f1} vs vaf1:{vaf1}\n")
    end_time = time.time()
    process_time6 = end_time - start_time
    # inference_time = process_time1 + process_time2 + process_time3 + process_time4 + process_time5 + process_time6
    inference_time = process_time6
    print(f"inference time: {inference_time}")
    return inference_time

def runOnlyFC(datasetname, result_path, aspect_id, aspect_cluster, maxiter=500, modeltype="", isabandon=False):
    if "LibSVM" in modeltype or "svm" in modeltype:
        params = getParams(datasetname, aspect_id, aspect_cluster)
    else:
        params = getParams(datasetname, aspect_id, aspect_cluster, modeltype=modeltype, othermodel=True)
    if len(params) == 0 :return
    cr = params['cr']
    issample = params['is_sample']
    bf = params['bf']
    isfc = params['isfc']
    isfc2 = params['isfc2']
    modeltype = params['model_type']
    svmparams = params['params']
    # acc = params['acc']
    # f1 = params['f1']
    solution = params['solution']
    elapsed_time = params['elapsedtime']
    if len(solution) < 1 : return
    filterscore = params['filter']
    data = Dataset(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster) #
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=issample)
    if len(train_data) == 0 or len(test_data) == 0:
        return
    x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, isabandon=isabandon)
    print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                (aspect_id, len(train_data), len(test_data)))
    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
    print(x_train.shape)
    scaler = Normalizer().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    fctraindata = x_train
    fctestdata = x_test
    if isfc:
        mafcdata = MafcData(name = datasetname) 
        mafcdata.fromArray(x_train)
        fctraindata = featureConstruction(mafcdata, isother = isfc2)
        
        mafcdata.fromArray(x_test)
        fctestdata = featureConstruction(mafcdata, isother = isfc2)
        print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
    fctraindataarray = np.array(fctraindata)
    fctestdataarray = np.array(fctestdata)
    if isfc or isfc2:
        fctraindataarray = np.nan_to_num(fctraindataarray)
        fctestdataarray = np.nan_to_num(fctestdataarray)
        scaler2 = Normalizer().fit(fctraindataarray)
        fctraindataarray = scaler2.transform(fctraindataarray)
        fctestdataarray = scaler2.transform(fctestdataarray)
    selectlist = []
    for i in range(len(fctraindataarray[0])):
        if normalized_mutual_info_score(fctraindataarray[:,i], y_train) > filterscore:
            selectlist.append(i)
    fctraindataarray = fctraindataarray[:,selectlist]
    fctestdataarray = fctestdataarray[:,selectlist]
    
    clf = None
    if "LibSVM" in modeltype:
        clf =  SVC(**svmparams, random_state=42, max_iter=maxiter)
    else:
        clf =  TSVC(**svmparams, random_state=42, max_iter=maxiter, n_jobs=8)
    X = fctraindataarray
    Y = y_train
    X_test = fctestdataarray
    
    print(f"train shape: {X.shape} test shape: {X_test.shape}\n")
    clf.fit(X, Y)
    pred = clf.predict(X_test)
    vaacc, vaf1 = evaluation(pred, y_test)
    clf_report = str(classification_report(y_test, pred))
    
    with open(result_path + 'svm_' + str(aspect_id), 'w') as f:
        f.write("################################################################\n")
        f.write("filter: " + str(filterscore) + "\n")
        f.write('model_type: ' + str(modeltype) + '\n')
        f.write('is_fc: ' + str(isfc) + '\n')
        f.write('is_fc2: ' + str(isfc2) + '\n')
        f.write('chi_ratio: ' + str(cr) + '\n')
        f.write('cr: ' + str(cr) + '\n')
        f.write('bow_features: ' + bf + '\n')
        f.write('is_sampling: ' + str(issample) + '\n')
        f.write(str(svmparams) + "\n")
        f.write('Optimized acc: %.5f \n' % vaacc)
        f.write('Optimized macro_f1: %.5f \n' % vaf1)
        f.write('training set shape: %s\n' % str(X.shape))
        f.write(clf_report)
        f.write(str(confusion_matrix(y_test, pred, labels=[-1,0,1]))+"\n")
        f.write("correct / total: %d / %d\n" % (sum(pred == y_test), len(y_test)))
        f.write("elapsed time: %.5f s\n" % elapsed_time)  

def runNoBalance(datasetname, result_path, aspect_id, aspect_cluster, maxiter=500, modeltype="", isabandon=False ,gpuid=0):

    if "now2v" not in modeltype and ("LibSVM" in modeltype or "svm" in modeltype):
        params = getParams(datasetname, aspect_id, aspect_cluster)
    else:
        params = getParams(datasetname, aspect_id, aspect_cluster, modeltype=modeltype, othermodel=True)
    if len(params) == 0 :return
    cr = params['cr']
    issample = params['is_sample']
    if issample is False:
        return
    issample = False
    bf = params['bf']
    isfc = params['isfc']
    isfc2 = params['isfc2']
    modeltype = params['model_type']
    svmparams = params['params']
    acc = params['acc']
    f1 = params['f1']
    solution = params['solution']
    filterscore = params['filter']
    start_time = time.time()
    data = Dataset(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster) #
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=issample)
    end_time = time.time()
    nobalance_time = end_time - start_time

    issample = True
    start_time = time.time()
    data = Dataset(base_dir=datasetname, is_preprocessed=True, ratio=cr, aspectcluster=aspect_cluster)  #
    train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=issample)
    end_time = time.time()
    balance_time = end_time - start_time

    return nobalance_time, balance_time
    # if len(train_data) == 0 or len(test_data) == 0:
    #     return
    # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, isabandon=isabandon)
    # print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
    #             (aspect_id, len(train_data), len(test_data)))
    # # x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf)
    # print(x_train.shape)
    # scaler = Normalizer().fit(x_train)
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    # fctraindata = x_train
    # fctestdata = x_test
    # if isfc:
    #     mafcdata = MafcData(name = datasetname)
    #     mafcdata.fromArray(x_train)
    #     fctraindata = featureConstruction(mafcdata, isother = isfc2)
    #
    #     mafcdata.fromArray(x_test)
    #     fctestdata = featureConstruction(mafcdata, isother = isfc2)
    #     print(f"fctraindata shape is {fctraindata.shape} fctestdata shape is {fctestdata.shape} ")
    # fctraindataarray = np.array(fctraindata)
    # fctestdataarray = np.array(fctestdata)
    # # listsolution = [1.0 for _ in range(fctraindataarray.shape[1])]
    # if isfc or isfc2:
    #     fctraindataarray = np.nan_to_num(fctraindataarray)
    #     fctestdataarray = np.nan_to_num(fctestdataarray)
    #     scaler2 = Normalizer().fit(fctraindataarray)
    #     fctraindataarray = scaler2.transform(fctraindataarray)
    #     fctestdataarray = scaler2.transform(fctestdataarray)
    # selectlist = []
    # for i in range(len(fctraindataarray[0])):
    #     if normalized_mutual_info_score(fctraindataarray[:,i], y_train) > filterscore:
    #         selectlist.append(i)
    # fctraindataarray = fctraindataarray[:,selectlist]
    # fctestdataarray = fctestdataarray[:,selectlist]
    #
    # clf = None
    # if "LibSVM" in modeltype:
    #     clf =  SVC(**svmparams, random_state=42, max_iter=maxiter)
    # else:
    #     clf =  TSVC(**svmparams, random_state=42, max_iter=maxiter, n_jobs=8, gpu_id=gpuid)
    # X = fctraindataarray
    # Y = y_train
    # X_test = fctestdataarray
    # # if len(solution) > 0:
    # #     selectlist = [i for i, n in enumerate(solution) if n == 1]
    # #     X = X[:,selectlist]
    # #     X_test = X_test[:,selectlist]
    #
    # print(f"train shape: {X.shape} test shape: {X_test.shape}\n")
    # clf.fit(X, Y)
    # pred = clf.predict(X_test)
    # vaacc, vaf1 = evaluation(pred, y_test)
    # clf_report = str(classification_report(y_test, pred))
    #
    # with open(result_path + 'svm_no_balance' + str(aspect_id), 'w') as f:
    #     f.write("################################################################\n")
    #     f.write("filter: " + str(filterscore) + "\n")
    #     f.write('model_type: ' + str(modeltype) + '\n')
    #     f.write('is_fc: ' + str(isfc) + '\n')
    #     f.write('is_fc2: ' + str(isfc2) + '\n')
    #     f.write('chi_ratio: ' + str(cr) + '\n')
    #     f.write('cr: ' + str(cr) + '\n')
    #     f.write('bow_features: ' + bf + '\n')
    #     f.write('is_sampling: ' + str(issample) + '\n')
    #     f.write(str(svmparams) + "\n")
    #     f.write('Optimized acc: %.5f \n' % vaacc)
    #     f.write('Optimized macro_f1: %.5f \n' % vaf1)
    #     f.write('training set shape: %s\n' % str(X.shape))
    #     f.write(clf_report)
    #     f.write(str(confusion_matrix(y_test, pred, labels=[-1,0,1]))+"\n")
    #     f.write("correct / total: %d / %d\n" % (sum(pred == y_test), len(y_test)))
    #     # f.write("elapsed time: %.5f s\n" % elapsed_time)


def main(datasetname, max_aspect_id=20, start_id=0, end_id=-1, execall=False, savepath="svm_results-k", \
    isfc=False, isrl=False, ispreprocessed=True, num_rounds2=2000, opttype=2,isthunder=False,isRL=False,\
        isVa=False, gpuid=0, igrate=0, maxiter=500, modeltype="thundersvm", isOnlyFC=False,\
            maxelapsetime=1, isabandon=False, ismatlibplot=False,isNoBalance=False):

    result_path = datasetname + f"optimal_results/{savepath}{max_aspect_id}/"
    if os.path.isdir(result_path):
        # remove_dirs(result_path)
        pass
    else: os.makedirs(result_path)
    chi_ratios = [x/10 for x in range(1, 11)]
    bow_features = ['all_words', 'parse_result', 'parse+chi']  #,'all_words',  'parse+chi'
    is_sampling = [True, False]
    best_accs = [-1 for _ in range(0, max_aspect_id)]
    best_f1 = [-1 for _ in range(0, max_aspect_id)]
    print(chi_ratios)
    num_rounds = 10
    scinstance = [0, 0]
    all_inference_time = 0
    all_nobalance_time = 0
    all_balance_time = 0
    for aspect_id in range(start_id, max_aspect_id):
        if end_id >= 0 and aspect_id > end_id:
            break 
        if isRL:
            runRL(datasetname, aspect_id, max_aspect_id, result_path, gpuid, maxiter, modeltype=modeltype, isabandon=isabandon)
            continue
        if isVa:

            inference_time = runVa(datasetname, aspect_id, max_aspect_id, maxiter, modeltype=modeltype, isabandon=isabandon, svandtraininstance=scinstance, gpuid=gpuid)
            if inference_time is not None:
                all_inference_time += inference_time
            continue
        if isOnlyFC:
            runOnlyFC(datasetname, result_path, aspect_id, max_aspect_id, maxiter, modeltype=modeltype, isabandon=isabandon)
            continue
        if isNoBalance:
            process_time = runNoBalance(datasetname, result_path, aspect_id, max_aspect_id, maxiter, modeltype=modeltype, isabandon=isabandon, gpuid=gpuid)
            if process_time is not None:
                all_nobalance_time += process_time[0]
                all_balance_time += process_time[1]
            continue
        # ht = HyperoptTunerLibSVM(maxiter=maxiter)
        # if isthunder:
        #     ht = HyperoptTuner(maxiter=maxiter)
        #     ht.gpu_id = gpuid
        ht = HyperoptTunerAll(maxiter=maxiter, modeltype=modeltype)
        ht.gpu_id = gpuid
        rlo = None
        if isrl:
            rlo = RLOptimization(dataname=datasetname, resultpath=result_path, aspect_id=aspect_id, \
                gpuid=gpuid, maxiter=maxiter)
        for bf in bow_features:
            if best_accs[aspect_id] == 1: break
            for iss in is_sampling:
                if best_accs[aspect_id] == 1: break
                if 'chi' in bf:
                    for cr in chi_ratios:
                        if best_accs[aspect_id] == 1: break
                        search(best_f1, aspect_id, bf, rlo, num_rounds, ht, num_rounds2, best_accs, result_path, \
                            issample=iss, cr=cr, ispreprocessed=ispreprocessed, aspect_cluster=max_aspect_id, \
                                opttype=opttype, isfc=isfc, igrate=igrate, maxelapsetime=maxelapsetime, \
                                    isabandon=isabandon, isrl=isrl, ismatlibplot=ismatlibplot)
                else:
                    search(best_f1, aspect_id, bf, rlo, num_rounds, ht, num_rounds2, best_accs, result_path, \
                        issample=iss, ispreprocessed=ispreprocessed, aspect_cluster=max_aspect_id, opttype=opttype, \
                            isfc=isfc, igrate=igrate, maxelapsetime=maxelapsetime, isabandon=isabandon,\
                                isrl=isrl, ismatlibplot=ismatlibplot)
    if isVa:
        print(f"svandtraininstance: {scinstance}")
        print(f"total inferece time: {all_inference_time}")
    if isNoBalance:
        print(f"all_nobalance_time: {all_nobalance_time}")
        print(f"all_balance_time: {all_balance_time}")
           
if __name__ == '__main__':

    # datasetnames = [LAPTOP_DIR, REST_DIR]
    # 定义短选项和长选项
    short_opts = "s:n:frp:e:am:Pi:t:TRVg:I:M:o:FE:blNh"
    long_opts = ["startid","endid","name", "featureconstruction", "reinforcementlearning", "savepath", \
            "all", "maxiternum", "Preprocessed", "iternums", "type", "Thunder", "ReinforcementLearing", "Validate",\
            "gpuid", "InformationGainrate", "Maxiternum", "modeltype", "OnlyFC", "Elapsedtime", "abandon", "matlibplot",\
            "NoBalance", "help"]

    # 解析命令行参数
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opts, long_opts)
    except getopt.GetoptError:
        print("Invalid command line arguments")
        sys.exit(2)

    # 处理选项
    gpuid = 0 #gpu id
    startid = 0 
    endid = -1 
    isfc = False 
    isrl = False 
    isthunder = False
    execall = False
    savepath = ""
    maxiternum = 20 #nums of aspect
    ispreprocessed = True
    iternums = 2000
    opttype=2
    isRL=False
    isVa=False
    igrate=0.1
    modeltype = "thundersvm"
    maxiter = 500 #svm max iter
    maxelapsetime = 1
    isOnlyFC = False
    isabandon = False
    ismatlibplot = False
    isNoBalance = False
    for opt, arg in opts:
        if opt in ("-s", "--startid"):
            startid = int(arg)
        elif opt in ("-n", "--name"):
            if arg == '0':
                datasetname = LAPTOP_DIR
            elif arg == '1':
                datasetname = REST_DIR
            elif arg == '2':
                datasetname = TWITTER_DIR
            elif arg == '3':
                datasetname = REST15_DIR
            elif arg == '4':
                datasetname = REST16_DIR
            elif arg == '5':
                datasetname = MAMS_DIR
        elif opt in ("-f", "--featureconstruction"):
            isfc = True
        elif opt in ("-r", "--reinforcementlearning"):
            isrl = True
        elif opt in ("-p", "--savepath"):
            savepath = arg
        elif opt in ("-e", "--endid"):
            endid = int(arg)
        elif opt in ("-a", "--all"):
            execall = True
        elif opt in ("-m", "--maxiternum"):
            maxiternum = int(arg)
        elif opt in ("-P", "--Preprocessed"):
            ispreprocessed = False
        elif opt in ("-i", "--iternums"):
            iternums = int(arg)
        elif opt in ("-t", "--type"):
            opttype = int(arg)
        elif opt in ("-T", "--Thunder"):
            isthunder = True
        elif opt in ("-R", "--ReinforcementLearning"):
            isRL = True
        elif opt in ("-V", "--Validate"):
            isVa = True
        elif opt in ("-g", "--gpuid"):
            gpuid = int(arg)
        elif opt in ("-I", "--InformationGainrate"):
            igrate = float(arg)
        elif opt in ("-M", "--Maxiternum"):
            maxiter = float(arg)
        elif opt in ("-o", "--modeltype"):
            modeltype = arg
        elif opt in ("-F", "--OnlyFC"):
            isOnlyFC = True
        elif opt in ("-E", "--Elapsedtime"):
            maxelapsetime = float(arg)
        elif opt in ("-b", "--abandon"):
            isabandon = True
        elif opt in ("-l", "--matlibplot"):
            ismatlibplot = True
        elif opt in ("-N", "--NoBalance"):
            isNoBalance = True
        elif opt in ("-h", "--help"):
            print("Usage: python whole_pipeline_ours.py [options]")
            print("Options:")
            print("-s, --startid: start id of aspect")
            print("-e, --endid: end id of aspect")
            print("-n, --name: dataset name")
            print("-f, --featureconstruction: feature construction")
            print("-r, --reinforcementlearning: reinforcement learning")
            print("-p, --savepath: save path")
            print("-a, --all: execute all")
            print("-m, --maxiternum: max iteration number")
            print("-P, --Preprocessed: is preprocessed")
            print("-i, --iternums: iteration number")
            print("-t, --type: type")
            print("-T, --Thunder: is thunder")
            print("-R, --ReinforcementLearning: is RL")
            print("-V, --Validate: is validate")
            print("-g, --gpuid: gpu id")
            print("-I, --InformationGainrate: information gain rate")
            print("-M, --Maxiternum: max iteration number")
            print("-o, --modeltype: model type")
            print("-F, --OnlyFC: only feature construction")
            print("-E, --Elapsedtime: elapsed time")
            print("-b, --abandon: abandon")
            print("-l, --matlibplot: matlibplot")
            print("-N, --NoBalance: NoBalance")
            print("-h, --help: help")
            sys.exit()

    if ispreprocessed is False:
        Dataset(base_dir=datasetname, is_preprocessed=ispreprocessed, aspectcluster=maxiternum)
        ispreprocessed = True
    # 处理参数
    # for arg in args:
    #     print("Processing file:", arg)

            
    print("dataset {0} begin".format(datasetname))
    main(datasetname, max_aspect_id=maxiternum, start_id=startid, end_id=endid, execall=execall, \
        savepath=savepath, isfc=isfc, isrl=isrl, ispreprocessed=ispreprocessed,num_rounds2=iternums,\
        opttype=opttype,isthunder=isthunder,isRL=isRL,isVa=isVa,gpuid=gpuid,igrate=igrate,\
            maxiter=maxiter, modeltype=modeltype, isOnlyFC=isOnlyFC, maxelapsetime=maxelapsetime,\
                isabandon=isabandon, ismatlibplot=ismatlibplot,isNoBalance=isNoBalance)
    print("dataset {0} end".format(datasetname))
    
    # datasetname = 'datasets/rest/'
    # main(datasetname)