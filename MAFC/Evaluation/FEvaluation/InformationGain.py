import copy
import math
from properties.properties import theproperty
import numpy as np
from dask_ml.model_selection import train_test_split
from MAFC_Operator.OperatorManager import OperatorManager
from Evaluation.FEvaluation.FEvaluation import FEvaluation
from logger.logger import logger
from Evaluation.Evaluation import evalutionScoringMethod
class InformationGainFilterEvaluator(FEvaluation):

    def __init__(self):
        super(InformationGainFilterEvaluator, self).__init__()
        self.valuesPerKey = {}

    def produceScore(self, DataDict, currentScore, oa, candidateAttributes):
        if candidateAttributes is not None:
            DataDict["data"][candidateAttributes[0]] = candidateAttributes[1]
        try:
            analyzedDatasets = copy.deepcopy(DataDict)
            om = OperatorManager()
            for acs in self.analycolumns:
                om.addColumn(analyzedDatasets, acs)
            bins = 10
            flag = super(InformationGainFilterEvaluator, self).discretizeColumns(analyzedDatasets, bins)
            self.valuesPerKey = {}
            X_train, X_test, y_train, y_test = train_test_split(analyzedDatasets["data"], analyzedDatasets["target"], random_state=theproperty.randomseed, test_size=0.3, shuffle=False)

            for cl in self.analycolumns:
                if theproperty.dataframe == "dask":
                    val = cl[1].compute().values
                elif theproperty.dataframe == "pandas":
                    val = cl[1].values
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")

                for index, value in zip(y_test.index, y_test):
                    indexlist = []
                    indexlist.append(val[index])
                    indexkey = tuple(indexlist)
                    if self.valuesPerKey.get(indexkey) is None:
                        numofunique = analyzedDatasets["targetInfo"].getNumsOfUnique()
                        if numofunique is None:
                            logger.Error(analyzedDatasets["targetInfo"].getName() + "Discrete Column is not exist numsofunique")
                        self.valuesPerKey[indexkey] = list(np.zeros(numofunique, dtype="int32"))
                    self.valuesPerKey[indexkey][value] += 1

            return self.calculateIG(X_train)
        except Exception as ex:
            logger.Error(f"InformationGain produceScore Error: {ex}", ex)
            return 0


    def calculateIG(self, dataset):
        try:
            IG = 0.0
            for val in self.valuesPerKey.values():
                numOfInstances = sum(val)
                tempIG = 0
                for va in val:
                    if va != 0:
                        tempIG += -((va / numOfInstances) * math.log10(va / numOfInstances))

                IG += (numOfInstances / len(dataset)) * tempIG
            return IG
        except Exception as ex:
            logger.Error(f"InformationGain calculateIG Error: {ex}", ex)
            return 0


    def needToRecalcScore(self) -> bool:
        return False

    def getEvalutionScoringMethod(self):
        return evalutionScoringMethod.InformationGain
