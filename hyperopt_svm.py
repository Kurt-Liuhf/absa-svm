from thundersvm import SVC
from hyperopt import hp, tpe, STATUS_OK, fmin
from sklearn.metrics import accuracy_score, f1_score, classification_report
from file_utils import *
import time
import os


class HyperoptTuner(object):

    def __init__(self, train_X=None, train_y=None, test_X=None, test_y=None, cluster_id=None, base_dir=None):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.cluster_id = cluster_id
        self.best_acc = .0
        self.best_f1 = .0
        self.best_iter = 0
        self.cnt = 0
        self.best_cfg = None
        self.clf_report = ""
        self.pred_results = []
        self.elapsed_time = None
        self.base_dir = base_dir
        self.correct = 0

    # pre-set parameters space
    def _preset_ps(self):
        space4svm = {
            'C': hp.uniform('C', 2 ** 10, 2 ** 20),
            'kernel': hp.choice('kernel', ['sigmoid', 'linear', 'rbf', 'polynomial']), #
            'gamma': hp.uniform('gamma', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            # 'gamma_value': hp.uniform('gamma_value', 0.001 / self.train_X.shape[1], 10.0 / self.train_X.shape[1]),
            'degree': hp.choice('degree', [i for i in range(1, 6)]),
            'coef0': hp.uniform('coef0', 1, 10)
        }

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
        params = self._svm_constraint(params)
        print("!!!!!!!!!!!!!!--->>> " + str(params))
        clf = SVC(**params, random_state=42)
        clf.fit(self.train_X, self.train_y)
        pred = clf.predict(self.test_X)
        self.pred_results = pred
        score_acc = accuracy_score(self.test_y, pred)
        score_f1 = f1_score(self.test_y, pred, average='macro')

        self.cnt += 1
        if score_acc > self.best_acc:
            self.best_acc = score_acc
            self.best_f1 = score_f1
            self.best_cfg = params
            self.best_iter = self.cnt
            self.clf_report = str(classification_report(self.test_y, pred))

        if is_tuning:
            print("****************************************************************")
            print('current params:\n  %s' % str(params))
            print('best params:\n  %s' % str(self.best_cfg))
            print('training set shape: %s' % str(self.train_X.shape))
            print('current acc / best acc: %.5f / %.5f' % (score_acc, self.best_acc))
            print('current_iter / best_iter: %d / %d' % (self.cnt, self.best_iter))
            print("best_macro_f1: %.5f" % self.best_f1)
            print(self.clf_report)
            print("****************************************************************")
        else:
            correct = 0
            for pred_y, true_y in zip(pred, self.test_y):
                if pred_y == true_y:
                    correct += 1
            self.correct = correct
            print("\n\n################################################################")
            print(params)
            print('Optimized acc: %.5f ' % score_acc)
            print('Optimized macro_f1: %.5f ' % score_f1)
            print(self.clf_report)
            print("####################################################################")
            make_dirs(os.path.join(self.base_dir, 'tmp_optimized_result'))
            path2save = os.path.join(self.base_dir, 'tmp_optimized_result', 'cluster_' + str(self.cluster_id))

            with open(path2save, 'w') as f:
                f.write("################################################################\n")
                f.write(str(params) + "\n")
                f.write('Optimized acc: %.5f \n' % score_acc)
                f.write('Optimized macro_f1: %.5f \n' % score_f1)
                f.write('training set shape: %s\n' % str(self.train_X.shape))
                f.write(self.clf_report)
                f.write("correct / total: %d / %d\n" % (correct, len(self.test_y)))
                f.write(str(self.elapsed_time) + "\n")
                f.write("################################################################")

        return score_acc

    def _object2minimize(self, params):
        score_acc = self._svm(params)
        return {'loss': 1 - score_acc, 'status': STATUS_OK}

    def tune_params(self, n_iter=200):
        t_start = time.time()
        fmin(fn=self._object2minimize,
            algo=tpe.suggest,
            space=self._preset_ps(),
            max_evals=n_iter)
        t_end = time.time()
        self.elapsed_time = t_end - t_start
        # print the final optimized result
        self._svm(self.best_cfg, is_tuning=False)

    def optimized_svm(self, params):
        self._svm(params, False)



