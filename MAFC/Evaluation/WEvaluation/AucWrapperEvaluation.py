from Evaluation.WEvaluation.WEvaluation import WEvaluation
from logger.logger import logger
from properties.properties import theproperty
from Evaluation.Evaluation import evalutionScoringMethod
from Evaluation.Classification.ClassificationResults import ClassificationResults


class AucWrapperEvaluation(WEvaluation):
    def __init__(self):
        pass

    def produceScore(self, analyzedatasets, currentscore: ClassificationResults, oa, candidateatt):
        try:
            if candidateatt is not None:

                analyzedatasets["data"][candidateatt[0]] = candidateatt[1]
            evaluationresults = self.runClassifier(theproperty.classifier, analyzedatasets)
            auc = self.calculateAUC(evaluationresults)
            if currentscore is not None:
                return auc - currentscore.getAuc()
            else:
                return auc
        except Exception as ex:
            logger.Error(f"WproduceScore error: {ex}", ex)
            return 0

    def getEvalutionScoringMethod(self):
        return evalutionScoringMethod.AUC