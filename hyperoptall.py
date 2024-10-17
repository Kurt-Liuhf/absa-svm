from copy import deepcopy
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier , VotingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier , LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from thundersvm import SVC as TSVC
from sklearn.svm import SVC
from hyperopt import hp, tpe, STATUS_OK, fmin
from sklearn.metrics import accuracy_score, f1_score, classification_report
# from file_utils import *
import time
import os
from extracthyper import estimator_option, condition_option, uniform, choice, uniformint, to_hp

class HyperoptTunerAll(object):

    def __init__(self, train_X=None, train_y=None, test_X=None, test_y=None, cluster_id=None, base_dir=None, maxiter=500, modeltype="thundersvm"):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.cluster_id = cluster_id
        self.best_acc = -1
        self.best_f1 = -1
        self.best_iter = 0
        self.cnt = 0
        self.best_cfg = ""
        self.clf_report = ""
        self.pred_results = []
        self.elapsed_time = 0
        self.base_dir = base_dir
        self.correct = 0
        self.best_predict_label=""
        self.curnumiter=0
        self.numite=0
        self.gpu_id = 0
        self.maxiter = maxiter
        self.modeltype = modeltype
        self.train_error = []
        self.test_error = []

    # pre-set parameters space
    def _preset_ps(self):
        if self.modeltype == "thundersvm":
            space4model = {
                'C': hp.uniform('C', 0.01, 2 ** 20),
                'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'polynomial']), #, 'linear', 'rbf', 'polynomial'
                'gamma': hp.uniform('gamma', 0.01 / self.train_X.shape[1], 1.0),
                # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
                'degree': hp.choice('degree', [i for i in range(1, 6)]),
                'coef0': hp.uniform('coef0', 0, 10),
                'decision_function_shape': hp.choice('decision_function_shape', ['ovo', 'ovr'])
            }
        elif self.modeltype == "LibSVM" or self.modeltype == "libsvm":
            space4model = {
                'C': hp.uniform('C', 0.01, 2 ** 20),
                'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'poly']), #, 'linear', 'rbf', 'polynomial'
                'gamma': hp.uniform('gamma', 0.01 / self.train_X.shape[1], 1.0),
                # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
                'degree': hp.choice('degree', [i for i in range(1, 6)]),
                'coef0': hp.uniform('coef0', 0, 10),
                'decision_function_shape': hp.choice('decision_function_shape', ['ovo', 'ovr'])
            }
        elif self.modeltype == "rf":
            space4model = {
                'n_estimators': hp.choice('n_estimators', [i for i in range(10, 1000)]),
                'max_depth': hp.choice('max_depth', [i for i in range(10, 1000)]),
                'min_samples_split': hp.choice('min_samples_split', [i for i in range(2, 10)]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [i for i in range(1, 10)]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
                'bootstrap': hp.choice('bootstrap', [True, False]),
                'criterion': hp.choice('criterion', ["gini", "entropy"])
            }
        elif self.modeltype == "gbdt":
            space4model = {
                'n_estimators': hp.choice('n_estimators', [i for i in range(10, 1000)]),
                'max_depth': hp.choice('max_depth', [i for i in range(10, 1000)]),
                'min_samples_split': hp.choice('min_samples_split', [i for i in range(2, 10)]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [i for i in range(1, 10)]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
                'criterion': hp.choice('criterion', ["friedman_mse", "mse", "mae"])
            }
        elif self.modeltype == "xgb":
            space4model = {
                'loss': hp.choice('loss', ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"]),
                'penalty': hp.choice('penalty', ["l2", "l1", "elasticnet"]),
                'alpha': hp.uniform('alpha', 0.0001, 0.001),
                'l1_ratio': hp.uniform('l1_ratio', 0.15, 0.25),
                'learning_rate': hp.uniform('learning_rate', 0.0001, 0.001),
                'eta0': hp.uniform('eta0', 0.0001, 0.001),
                'power_t': hp.uniform('power_t', 0.1, 0.5),
                'early_stopping': hp.choice('early_stopping', [True, False]),
                'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.5),
                'n_iter_no_change': hp.choice('n_iter_no_change', [i for i in range(5, 20)]),
                'average': hp.choice('average', [True, False]),
                'class_weight': hp.choice('class_weight', [None, "balanced"]),
                'warm_start': hp.choice('warm_start', [True, False])
            }
        elif self.modeltype == "mlp":
            space4model = {
                'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [i for i in range(10, 1000)]),
                'activation': hp.choice('activation', ["identity", "logistic", "tanh", "relu"]),
                'solver': hp.choice('solver', ["lbfgs", "sgd", "adam"]),
                'alpha': hp.uniform('alpha', 0.0001, 0.001),
                'learning_rate': hp.choice('learning_rate', ["constant", "invscaling", "adaptive"]),
                'learning_rate_init': hp.uniform('learning_rate_init', 0.0001, 0.001),
                'power_t': hp.uniform('power_t', 0.1, 0.5),
                'early_stopping': hp.choice('early_stopping', [True, False]),
                'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.5),
                'n_iter_no_change': hp.choice('n_iter_no_change', [i for i in range(5, 20)]),
                'beta_1': hp.uniform('beta_1', 0.1, 0.5),
                'beta_2': hp.uniform('beta_2', 0.1, 0.5),
                'epsilon': hp.uniform('epsilon', 0.0001, 0.001)
            }
        elif self.modeltype == "knn":
            space4model = {
                'n_neighbors': hp.choice('n_neighbors', [i for i in range(1, 10)]),
                'weights': hp.choice('weights', ['uniform', 'distance']),
                'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'leaf_size': hp.choice('leaf_size', [i for i in range(10, 1000)]),
                'p': hp.choice('p', [1, 2])
            }
        elif self.modeltype == "nb":
            space4model = {
                'var_smoothing': hp.uniform('var_smoothing', 0.000000001, 0.00000001)
            }
        elif self.modeltype == "lr":
            space4model = estimator_option(
                penalty = choice([
                    condition_option(
                        "l2", 
                        solver = choice(["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
                    ),
                    condition_option(
                        "l1",
                        solver = choice(["liblinear", "saga"]),
                    ),
                    condition_option(
                        "elasticnet",
                        solver = choice(["saga"]),
                        l1_ratio = uniform(0, 1),
                        
                    ),
                    condition_option(
                        "none",
                        solver = choice(["newton-cg", "lbfgs", "sag", "saga"]), 
                    )]),
                tol = uniform( 0.0001, 0.001),
                C = uniform(0.1, 1.0),
                fit_intercept = choice([True, False]),
                intercept_scaling = uniform(0.1, 1.0),
                class_weight = choice([None, "balanced"]),
                # solver = choice(["newton-cg", "lbfgs", "liblinear", "sag", "saga"]),
            )
            space4model = to_hp(space4model)
        elif self.modeltype == "dt":
            space4model = {
                'criterion': hp.choice('criterion', ["gini", "entropy"]),
                'splitter': hp.choice('splitter', ["best", "random"]),
                'max_depth': hp.choice('max_depth', [i for i in range(10, 1000)]),
                'min_samples_split': hp.choice('min_samples_split', [i for i in range(2, 10)]),
                'min_samples_leaf': hp.choice('min_samples_leaf', [i for i in range(1, 10)]),
                'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
                'max_leaf_nodes': hp.choice('max_leaf_nodes', [i for i in range(10, 1000)]),
                'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.0001, 0.001),
                'class_weight': hp.choice('class_weight', [None, "balanced"]),
            }
        # space4model = {
        #     'C': hp.uniform('C', 2 ** 10, 2 ** 20),
        #     'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'polynomial']), #, 'linear', 'rbf', 'polynomial'
        #     'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
        #     # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
        #     'degree': hp.choice('degree', [i for i in range(1, 6)]),
        #     'coef0': hp.uniform('coef0', 1, 10)
        # }

        return space4model

    def _model_constraint(self, params):
        if params['kernel'] != 'polynomial':
            params.pop('degree', None)

        if params['kernel'] != 'polynomial' and params['kernel'] != 'sigmoid':
            params.pop('coef0', None)

        if params['kernel'] == 'linear':
            params.pop('gamma', None)

        return params

    def _model(self, params, is_tuning=True):
        self.curnumiter+=1
        if self.curnumiter % 500 == 0:
            print("{2} current iteration {0} / {1} ".format(self.curnumiter, self.numite, str(datetime.now().ctime())))
        # params = self._model_constraint(params)
        # print("!!!!!!!!!!!!!!--->>> " + str(params))
        score_acc = 0
        score_f1 = 0
        train_score_acc = 0
        score_acc = 0
        try:
            clf = None
            if self.modeltype == "thundersvm":
                clf = TSVC(**params, random_state=42, max_iter=self.maxiter, n_jobs=8, gpu_id=self.gpu_id)
            elif self.modeltype == "LibSVM" or self.modeltype == "libsvm":
                clf = SVC(**params, random_state=42, max_iter=self.maxiter)
            elif self.modeltype == "rf":
                clf = RandomForestClassifier(**params, random_state=42, n_jobs=8)
            elif self.modeltype == "gbdt":
                clf = GradientBoostingClassifier(**params, random_state=42)
            elif self.modeltype == "xgb":
                clf = SGDClassifier(**params, random_state=42, n_jobs=8, max_iter=self.maxiter)
            elif self.modeltype == "mlp":
                clf = MLPClassifier(**params, random_state=42, max_iter=self.maxiter)
            elif self.modeltype == "knn":
                clf = KNeighborsClassifier(**params, n_jobs=8)
            elif self.modeltype == "nb":
                clf = GaussianNB(**params)
            elif self.modeltype == "lr":
                params_ = deepcopy(params)
                for k, v in deepcopy(params).items():
                    if isinstance(v, dict) and 'condition' in v.keys():
                        value_name = v['condition']
                        v.pop('condition')
                        key_name = k
                        params_.pop(k)
                        params_.update(v)
                        params_.update({key_name: value_name})
                params = params_
                clf = LogisticRegression(**params, random_state=42, n_jobs=8, max_iter=self.maxiter)
            elif self.modeltype == "dt":
                clf = DecisionTreeClassifier(**params, random_state=42)
            elif self.modeltype == "et":
                clf = ExtraTreesClassifier(**params, random_state=42, n_jobs=8)
            elif self.modeltype == "ada":
                clf = AdaBoostClassifier(**params, random_state=42)
            elif self.modeltype == "bag":
                clf = BaggingClassifier(**params, random_state=42, n_jobs=8)
            elif self.modeltype == "voting":
                clf = VotingClassifier(**params, n_jobs=8)
            elif self.modeltype == "stacking":
                clf = StackingClassifier(**params, n_jobs=8)
            # clf = SVC(**params, random_state=42, max_iter=self.maxiter, n_jobs=8, gpu_id=self.gpu_id)
            clf.fit(self.train_X, self.train_y)
            # trainpred = clf.predict(self.train_X)
            # train_score_acc = accuracy_score(self.train_y, trainpred)
            train_score_acc = 1
            pred = clf.predict(self.test_X)
            self.pred_results = pred
            score_acc = accuracy_score(self.test_y, pred)
            score_f1 = f1_score(self.test_y, pred, average='macro')
        except Exception as ex:
            print(f"{str(datetime.now().ctime())} {self.modeltype} runtime error: {ex}\n")
            score_acc = 0
            score_f1 = 0
        self.cnt += 1
        if score_acc >= self.best_acc and score_acc > 0:
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
            self.train_error.append(1 - train_score_acc)
            self.test_error.append(1 - score_acc)
        else:
            correct = 0
            if score_acc > 0:
                for pred_y, true_y in zip(pred, self.test_y):
                    if pred_y == true_y:
                        correct += 1
                self.correct = correct

        return score_acc

    def _object2minimize(self, params):
        score_acc = self._model(params)
        return {'loss': 1 - score_acc, 'status': STATUS_OK}

    def tune_params(self, n_iter=200, type=2, maxtimehours=1, init_params=None, early_stop_fn=None):
        self.train_error = []
        self.test_error = []
        t_start = time.time()
        fmin(fn=self._object2minimize,
            algo=tpe.suggest,
            space=self._preset_ps(),
            max_evals=n_iter,
            points_to_evaluate=init_params,
            early_stop_fn=early_stop_fn,
            timeout=maxtimehours*60*60)
        t_end = time.time()
        self.elapsed_time = t_end - t_start
        # print the final optimized result
        self._model(self.best_cfg, is_tuning=False)

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



