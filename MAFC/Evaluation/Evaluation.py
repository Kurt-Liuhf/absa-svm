from abc import abstractmethod
from enum import Enum
from logger.logger import logger
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
class evalutionType(Enum):
    Filte = 0
    Wrapper = 1


class evalutionScoringMethod(Enum):
    AUC = 0
    InformationGain = 1
    Fone = 2
    LogLoss = 3
    ClassifierProbability = 4


class Evaluation:
    def __init__(self):
        self.evalutionType = evalutionType
        self.evalutionScoringMethod = evalutionScoringMethod

    @abstractmethod
    def produceScore(self, Datadict, currentScore, oa, candidateAttribute):
        pass

    @abstractmethod
    def getType(self):
        pass

    @abstractmethod
    def getEvalutionScoringMethod(self):
        pass

    def getClassifier(self, classifiername):
        model = None
        if classifiername == "RandomForest":
            model = RandomForestClassifier()
        elif classifiername == "DicisionTree":
            model = DecisionTreeClassifier()
        elif classifiername == "SVM":
            model = SVC(probability=True)
        else:
            logger.Error("No this Model : " + classifiername)
        return model


