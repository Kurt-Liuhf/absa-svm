import copy
import os
from datetime import datetime
from random import random

import numpy as np

from MAFC.Serialize import serialize, deserialize
from logger.logger import logger
from Evaluation.Evaluation import *
from Evaluation.Classification.ClassificationResults import ClassificationResults
from properties.properties import theproperty
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, label_binarize
from dask_ml.wrappers import ParallelPostFit
from dask_ml.model_selection import train_test_split
from distributed import Client
from MAFC_Operator.Operators import Operators

class WEvaluation(Evaluation):
    def __init__(self):
        super(WEvaluation, self).__init__()

    def evaluationAsave(self, currentresult: ClassificationResults, iteration=0, addatts=None, nums=0, newfile=True, istest = False):
        try:
            sb = ''
            if newfile:
                sb += "Iteration,Added_Attributes,LogLoss,AUC,F1Score"
                sb += ",Chosen_Attributes_FScore,Chosen_Attributes_WScore,Num_Of_Evaluated_Attributes_In_Iteration"
                sb += ",Iteration_Completion_time"
                sb += os.linesep

            sb += str(iteration) + ","
            sb += '"['
            if addatts is not None:
                for ats in addatts:
                    sb += ats.getName()
                    sb += "+"
            sb += ']",'
            sb += str(round(currentresult.getLogLoss(), 2)) + ","
            sb += str(round(currentresult.getAuc(), 4)) + ','
            sb += str(round(currentresult.getFMeasureValues(), 4)) + ","
            sb += '"['
            if addatts is not None:
                for ats in addatts:
                    sb += str(round(ats.getFScore(), 4))
                    sb += ","
                sb += ']","['
                for ats in addatts:
                    sb += str(round(ats.getWScore(), 4))
                    sb += ","
                sb += ']",'
            else:
                sb += ']","[]",'
            sb += str(nums) + ","
            date = datetime.now()
            sb += date.__str__().replace(" ", "")

            if istest == False:
                filepath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "result.csv";
            else:
                filepath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "testresult.csv";
            if newfile:
                fw = open(filepath, "w")
            else:
                fw = open(filepath, "a")
            fw.write(sb + "\n")

        except Exception as ex:
            logger.Error(f"IOException: {ex}", ex)
        finally:
            fw.close()

    def ProduceClassifications(self, dataset, classifier):
        datacopy = copy.deepcopy(dataset)
        classificationresult = self.getClassifications(datacopy, classifier)
        return classificationresult

    def runClassifier(self, classifiername, datadict):
        try:
            if theproperty.dataframe == "dask":
                client = Client()
            copydata = datadict["data"].copy()
            y = datadict["target"]
            if y.name in copydata.columns:
                del copydata[y.name]
            for name in copydata:
                if copydata[name].dtype not in ["int32", "int64", "float32", "float64", "bool"]:
                    del copydata[name]
            model = self.getClassifier(classifiername)
            res = None
            X_train, X_test, y_train, y_test = train_test_split(copydata, y, test_size=0.3, random_state=theproperty.randomseed)
            #lens = (len(copydata), len(X_train), len(X_test), len(y_train), len(y_test))
            i = 0
            if theproperty.dataframe == "dask":
                y_test_un = len(np.unique(y_test.values.compute()))
                y_train_un = len(np.unique(y_train.values.compute()))
            elif theproperty.dataframe == "pandas":
                y_test_un = len(np.unique(y_test.values))
                y_train_un = len(np.unique(y_train.values))
            if y_test_un != theproperty.targetclasses or y_train_un != theproperty.targetclasses:
                while i < 100000:
                    seed = int(random() * 100000)

                    X_train, X_test, y_train, y_test = train_test_split(copydata, y, test_size=0.3,
                                                                        random_state=seed)
                    i += 1
                    if theproperty.dataframe == "dask":
                        y_test_un = len(np.unique(y_test.values.compute()))
                        y_train_un = len(np.unique(y_train.values.compute()))
                    elif theproperty.dataframe == "pandas":
                        y_test_un = len(np.unique(y_test.values))
                        y_train_un = len(np.unique(y_train.values))
                    #print(seed, y_test_un, y_train_un, theproperty.targetclasses)
                    if y_test_un == theproperty.targetclasses and y_train_un == theproperty.targetclasses:
                        theproperty.randomseed = seed
                        logger.Info(f"find the seed{seed}")
                        break

            if theproperty.dataframe == "dask":
                clf = ParallelPostFit(model, scoring="r2")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_pro = clf.predict_proba(X_test)
                y_p = y_pred.compute()
                y_t = y_test.compute().values
                y_pro = y_pred_pro.compute()
            elif theproperty.dataframe == "pandas":
                model.fit(X_train, y_train)
                y_p = model.predict(X_test)
                y_t = y_test.values
                y_pro = model.predict_proba(X_test)
            #y_t = [val[1] for val in y_t]
            res = [y_t, y_p, y_pro]

        except Exception as ex:
            #logger.Error(f"runclassifier error{ex}", ex)
            pass

        finally:
            if theproperty.dataframe == "dask":
                client.close()
            return res

    def calculateAUC(self, evaluation):
        if evaluation is None:
            return 0
        try:
            multi = theproperty.targetmutil
            if multi:
                # y_one_hot = label_binarize(evaluation[0], classes=np.arange(theproperty.targetclasses))
                # ytrue_one_hot = label_binarize(evaluation[1], classes=np.arange(theproperty.targetclasses))
                auc = roc_auc_score(evaluation[0], evaluation[2], multi_class='ovr')
            else:
                auc = roc_auc_score(evaluation[0], evaluation[1])
            return auc
        except ValueError as ex:
            logger.Error(f"calculateAUC error {ex}", ex)
            return 0

    def calculateLoss(self, evaluation):
        if evaluation is None:
            return 0
        try:
            # one_hot = OneHotEncoder(sparse=False)
            # y_true = [[i] for i in evaluation[0]]
            # y_true = one_hot.fit_transform(y_true)
            # y_pred = [[i] for i in evaluation[1]]
            # y_pred = one_hot.fit_transform(y_pred)
            return log_loss(evaluation[0], evaluation[2])
        except Exception as ex:
            logger.Error(f"calculateLoss error{ex}")
            return 0


    def calculateFsocre(self, evaluation):
        if evaluation is None:
            return 0
        try:
            return f1_score(evaluation[0], evaluation[1], average='macro')
        except Exception as ex:
            logger.Error(f"calculateFsocre error{ex}", ex)
            return 0

    def getClassifications(self, datadict, classifier):
        evaluations = self.runClassifier(classifier, datadict)
        auc = self.calculateAUC(evaluations)
        loss = self.calculateLoss(evaluations)
        fmeasurevalue = self.calculateFsocre(evaluations)
        return ClassificationResults(auc, loss, fmeasurevalue)

    def saveModel(self, datadictori, classifiername, iters):
        datadict = copy.deepcopy(datadictori)
        try:
            if theproperty.dataframe == "dask":
                client = Client()
            copydata = datadict["data"].copy()
            y = datadict["target"]
            if y.name in copydata.columns:
                del copydata[y.name]
            for name in copydata:
                if copydata[name].dtype not in ["int32", "int64", "float32", "float64", "bool"]:
                    del copydata[name]
            model = self.getClassifier(classifiername)
            res = None
            modelpath = theproperty.rootpath + theproperty.backmodelpath + theproperty.dataframe + theproperty.datasetname + classifiername + str(iters) + ".model"
            if theproperty.dataframe == "dask":
                clf = ParallelPostFit(model, scoring="r2")
                clf.fit(copydata, y)
                serialize(modelpath, clf)
            elif theproperty.dataframe == "pandas":
                model.fit(copydata, y)
                serialize(modelpath, model)
        except Exception as ex:
            logger.Error(f"saveModel error{ex}", ex)

        finally:
            if theproperty.dataframe == "dask":
                client.close()

    def loadModal(self, classifiername, iters):
        modelpath = theproperty.rootpath + theproperty.backmodelpath + theproperty.dataframe + theproperty.datasetname + classifiername + str(iters) + ".model"
        return deserialize(modelpath)

    def getTestClassifications(self, datadictori, classifier, iters):
        try:
            model = self.loadModal(classifier, iters)
            datadict = copy.deepcopy(datadictori)
            if theproperty.dataframe == "dask":
                client = Client()
            copydata = datadict["data"].copy()
            y = datadict["target"]
            if y.name in copydata.columns:
                del copydata[y.name]
            for name in copydata:
                if copydata[name].dtype not in ["int32", "int64", "float32", "float64", "bool"]:
                    del copydata[name]
            if theproperty.dataframe == "dask":
                y_pred = model.predict_proba(copydata)
                y_pred_pro = model.predict_proba(copydata)
                y_p = y_pred.compute()
                y_t = y.compute().values
                y_pro = y_pred_pro.compute()
            elif theproperty.dataframe == "pandas":

                y_p = model.predict(copydata)
                y_t = y.values
                y_pro = model.predict_proba(copydata)
            evaluations = [y_t, y_p, y_pro]
            auc = self.calculateAUC(evaluations)
            loss = self.calculateLoss(evaluations)
            fmeasurevalue = self.calculateFsocre(evaluations)
            return ClassificationResults(auc, loss, fmeasurevalue)
        except Exception as ex:
            logger.Error(f"getTestClassifications error{ex}", ex)
            return None

        finally:
            if theproperty.dataframe == "dask":
                client.close()





