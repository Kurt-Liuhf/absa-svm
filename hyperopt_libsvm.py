from hyperopt import hp, tpe, STATUS_OK, fmin
from sklearn.metrics import accuracy_score, f1_score, classification_report
from file_utils import *
from sklearn.svm import SVC
import time
import os
from datetime import datetime
from ultraopt import fmin as ulfmin

class HyperoptTunerLibSVM(object):

    def __init__(self, train_X=None, train_y=None, test_X=None, test_y=None, cluster_id=None, base_dir=None, maxiter=500):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.cluster_id = cluster_id
        self.best_acc = -1
        self.best_f1 = -1
        self.best_iter = 0
        self.cnt = 0
        self.best_cfg = None
        self.clf_report = ""
        self.pred_results = []
        self.elapsed_time = None
        self.base_dir = base_dir
        self.correct = 0
        self.best_predict_label=None
        self.curnumiter=0
        self.numite=0
        self.maxiter = maxiter
        
    # pre-set parameters space
    def _preset_ps(self):
        space4svm = {
            'C': hp.uniform('C', 2 ** 10, 2 ** 20),
            'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'poly']), #, 'linear', 'rbf', 'polynomial'
            'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            'degree': hp.choice('degree', [i for i in range(1, 6)]),
            'coef0': hp.uniform('coef0', 1, 10)
        }

        return space4svm

    def _Ulpreset_ps(self):
        hdl = {
            'C': {"_type":"uniform", "_value":[2 ** 10, 2 ** 20]},#hp.uniform('C', 2 ** 10, 2 ** 20),
            'kernel': {"_type":"choice", "_value":['sigmoid', 'linear', 'rbf', 'poly']},#'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'poly']), #, 'linear', 'rbf', 'polynomial'
            'gamma': {"_type":"uniform", "_value":[0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]]},#'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            'degree': {"_type":"int_uniform", "_value":[2,5]}, #'degree': hp.choice('degree', [i for i in range(1, 6)]),
            'coef0': {"_type":"uniform", "_value":[1,10]}, #'coef0': hp.uniform('coef0', 1, 10)
        }
        return hdl
    
    def _svm_constraint(self, params):
        # if params['kernel'] != 'poly':
        #     params.pop('degree', None)

        if params['kernel'] != 'poly' and params['kernel'] != 'sigmoid':
            params.pop('coef0', None)

        if params['kernel'] == 'linear':
            params.pop('gamma', None)

        return params

    def _svm(self, params, is_tuning=True):
        self.curnumiter+=1
        if self.curnumiter % 500 == 0:
            print("{2} current iteration {0} / {1} ".format(self.curnumiter, self.numite, str(datetime.now().ctime())))
        # params = self._svm_constraint(params)
        # print("!!!!!!!!!!!!!!--->>> " + str(params))
        score_acc = 0
        score_f1 = 0
        try:
            clf = SVC(**params, random_state=42, max_iter=self.maxiter)
            clf.fit(self.train_X, self.train_y)
            pred = clf.predict(self.test_X)
            self.pred_results = pred
            score_acc = accuracy_score(self.test_y, pred)
            score_f1 = f1_score(self.test_y, pred, average='macro')
        except Exception as ex:
            print(f"{str(datetime.now().ctime())} svm runtime error: {ex}\n")
            score_acc = 0
            score_f1 = 0

        self.cnt += 1
        if score_acc >= self.best_acc:
            if score_acc > self.best_acc or score_f1 > self.best_f1:
                # print("{1} best params:\n {0}".format(str(self.best_cfg), str(datetime.now().ctime())))
                # print("{1} find best ,current params:\n {0}".format(str(params), str(datetime.now().ctime())))
                self.best_acc = score_acc
                self.best_f1 = score_f1
                self.best_cfg = params
                self.best_iter = self.cnt
                self.clf_report = str(classification_report(self.test_y, pred))
                self.best_predict_label = self.pred_results
                print("{1} find best acc: {2} f1: {3},current params:\n {0}".format(str(params), str(datetime.now().ctime()), self.best_acc, self.best_f1))
                
        if is_tuning:
            pass
        else:
            correct = 0
            for pred_y, true_y in zip(pred, self.test_y):
                if pred_y == true_y:
                    correct += 1
            self.correct = correct

        return score_acc

    def _object2minimize(self, params):
        score_acc = self._svm(params)
        return {'loss': 1 - score_acc, 'status': STATUS_OK}

    def _ulobject2minimize(self, params):
        score_acc = self._svm(params)
        return 1 - score_acc

    def tune_params(self, n_iter=2000, type=2, maxtimehours=1):
        t_start = time.time()
        self.curnumiter=0
        self.numite=n_iter
        if type == 1:
            fmin(fn=self._object2minimize,
                algo=tpe.suggest,
                space=self._preset_ps(),
                max_evals=n_iter,
                show_progressbar=False,
                timeout=maxtimehours*60*60)
        elif type == 2:
            ulfmin(eval_func=self._ulobject2minimize,
                optimizer="ETPE",
                config_space=self._Ulpreset_ps(),
                n_iterations=n_iter)
                # show_progressbar=True)
        t_end = time.time()
        self.elapsed_time = t_end - t_start
        # print the final optimized result
        self._svm(self.best_cfg, is_tuning=False)

    def optimized_svm(self, params):
        self._svm(params, False)
        
    def getAcc(self, traindata, trainlabel, testdata, testlabel):
        params = self.best_cfg
        # print("!!!!!!!!!!!!!!--->>> " + str(params))
        clf = SVC(**params, random_state=42)
        clf.fit(traindata, trainlabel)
        pred = clf.predict(testdata)
        pred_results = pred
        score_acc = accuracy_score(self.test_y, pred)
        score_f1 = f1_score(self.test_y, pred, average='macro')

        correct = 0
        for pred_y, true_y in zip(pred, self.test_y):
            if pred_y == true_y:
                correct += 1
        

        print('Original data Optimized acc: %.5f \n' % score_acc)
        print('Original Optimized macro_f1: %.5f \n' % score_f1)    
        print('training set shape: %s\n' % str(self.train_X.shape))
        print("correct / total: %d / %d\n" % (correct, len(self.test_y)))

        return score_acc
    
