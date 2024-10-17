import copy, time
from qbso_fs.solution import Solution
from qbso_fs.rl import QLearning
from datetime import datetime
from qbso_fs.fs_problem import FsProblem
from qbso_fs.swarm import Swarm
from sklearn.svm import SVC
from thundersvm import SVC as TSVC
from hyperopt import hp, tpe, STATUS_OK, fmin
from sklearn.metrics import accuracy_score, f1_score, classification_report
from file_utils import *
import os
# RL 

alpha = 0.1
gamma = 0.99
epsilon = 0.01

# BSO

flip = 4
max_chance = 3
bees_number = 20
maxIterations = 20
locIterations = 100

# Test type

typeOfAlgo = 1
nbr_exec = 1
method = "qbso_simple"
test_param = "rl"
param = "gamma"
val = str(locals()[param])


class RLOptimization(object):
    
    def __init__(self, train_X=None, train_y=None, test_X=None, test_y=None, dataname = None, resultpath = None, params = None, aspect_id=None, gpuid = 0, maxiter=500):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.dataname = dataname
        self.params = params
        if params is not None:
            self.param = params['params']
            self.modelname = params['model_type']
        else:
            self.param = None
            self.modelname = "thundersvm"
        self.result_path = resultpath
        self.aspect_id = aspect_id
        self.best_acc = -1
        self.best_f1 = -1
        self.clf_report = None
        self.best_predict_label = None
        if self.param is not None:
            self.best_cfg = self.param
        self.solution = []
        self.typeOfAlgo = typeOfAlgo
        self.nb_exec = nbr_exec
        self.numsfea = 0
        self.correct = 0
        self.pred_results = []
        self.elapsed_time = 0
        self.curnumiter=0
        self.numite=0
        self.maxiter=maxiter
        if self.train_X is not None and len(self.train_X) > 0:
            self.numsfea = len(self.train_X[0])
        self.ql = QLearning(self.numsfea+1, Solution.attributs_to_flip(self.numsfea), alpha, gamma, epsilon)
        self.fsd = None
        self.gpuid = gpuid
        self.cnt = 0
        
    def run(self):
        t_init = time.time()
        total_time = 0
        for itr in range(1,self.nb_exec+1):
            print ("Execution {0}".format(str(itr)))
            self.fsd = FsProblem(self.typeOfAlgo, self.train_X, self.train_y, self.test_X, self.test_y, self.ql, self.numsfea, self.param, self.modelname, self.gpuid, self.maxiter)
            swarm = Swarm(self.fsd, flip, max_chance, bees_number, maxIterations, locIterations)
            t1 = time.time()
            best = swarm.bso(self.typeOfAlgo, flip, self)
            t2 = time.time()
            total_time += t2-t1
                
        t_end = time.time()
        self.elapsed_time = t_end-t_init
        print ("{2} Total execution time for dataset {0} is {1:.2f} s".format(self.dataname, t_end-t_init, str(datetime.now().ctime())))
        

    # pre-set parameters space
    def _preset_ps(self):
        space4svm = {
            'C': hp.uniform('C', 2 ** 10, 2 ** 20),
            'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'polynomial']), #, 'linear', 'rbf', 'polynomial'
            'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            'degree': hp.choice('degree', [i for i in range(1, 6)]),
            'coef0': hp.uniform('coef0', 1, 10)
        }
        for i in range(len(self.train_X[0])):
            space4svm[str(i)] = hp.choice(str(i), [0, 1])

        return space4svm

    def _svm_constraint(self, params):
        if params['kernel'] != 'polynomial':
            params.pop('degree', None)

        if params['kernel'] != 'polynomial' and params['kernel'] != 'sigmoid':
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
        clf = None
        score_acc = 0
        score_f1 = 0
        svmparams={"C": params["C"], "kernel": params["kernel"], "gamma": params["gamma"],\
                "degree": params["degree"], "coef0": params["coef0"]}
        try:
            if "LibSVM" in self.modelname: 
                clf = SVC(**svmparams, random_state=42, max_iter=self.maxiter)
            else: clf = TSVC(**svmparams, random_state=42, max_iter=self.maxiter, n_jobs=8, gpu_id=self.gpuid)
            selectlist = []
            for i in range(len(self.train_X[0])):
                val = params.get(str(i))
                if val > 0:
                    selectlist.append(i)
            if len(selectlist) == 0:
                return 0
            X = self.train_X[:,selectlist]
            Y = self.train_y
            Xtest = self.test_X[:,selectlist]
            
            clf.fit(X, Y)
            pred = clf.predict(Xtest)
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
                
                self.best_acc = score_acc
                self.best_f1 = score_f1
                self.best_cfg = svmparams
                self.best_iter = self.cnt
                self.clf_report = str(classification_report(self.test_y, pred))
                self.best_predict_label = self.pred_results
                self.solution = [0 for _ in range(len(self.train_X[0]))]
                for i in selectlist:
                    self.solution[i] = 1
                # print("{1} best params:\n {0}".format(str(self.best_cfg), str(datetime.now().ctime())))
                print("{1} find best acc: {2} f1: {3},current params:\n {0}".format(str(svmparams), str(datetime.now().ctime()), self.best_acc, self.best_f1))
        
                correct = 0
                for pred_y, true_y in zip(pred, self.test_y):
                    if pred_y == true_y:
                        correct += 1
                self.correct = correct

        return score_acc

    def _object2minimize(self, params):
        score_acc = self._svm(params)
        return {'loss': 1 - score_acc, 'status': STATUS_OK}

    def tune_params(self, n_iter=2000, type=2, maxtimehours=3):
        t_start = time.time()
        fmin(fn=self._object2minimize,
            algo=tpe.suggest,
            space=self._preset_ps(),
            max_evals=n_iter,
            timeout=maxtimehours*60*60)
        t_end = time.time()
        self.elapsed_time = t_end - t_start
        # print the final optimized result
        # self._svm(self.best_cfg, is_tuning=False)

    def optimized_svm(self, params):
        self._svm(params, False)
        



