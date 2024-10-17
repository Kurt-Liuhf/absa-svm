from Evaluation.WEvaluation.WEvaluation import WEvaluation
from properties.properties import theproperty
from Evaluation.Evaluation import evalutionScoringMethod
from Evaluation.Classification.ClassificationResults import ClassificationResults


class LogLossEvaluation(WEvaluation):
    def __init__(self):
        pass

    def produceScore(self, analyzedatasets, currentscore: ClassificationResults, oa, candidateatt):
        if candidateatt is not None:
            #留坑
            analyzedatasets["data"][candidateatt[0]] = candidateatt[1]
        evaluationresults = self.runClassifier(theproperty.classifier, analyzedatasets)
        loss = self.calculateLoss(evaluationresults)
        if currentscore is not None:
            return loss - currentscore.getLogLoss()
        else:
            return loss

    def getEvalutionScoringMethod(self):
        return evalutionScoringMethod.LogLoss