from Evaluation.WEvaluation.WEvaluation import WEvaluation
from logger.logger import logger
from properties.properties import theproperty
from Evaluation.Evaluation import evalutionScoringMethod
from Evaluation.Classification.ClassificationResults import ClassificationResults


class FoneEvaluation(WEvaluation):
    def __init__(self):
        pass

    def produceScore(self, analyzedatasets, currentscore: ClassificationResults, oa, candidateatt):
        try:
            if candidateatt != None:

                analyzedatasets["data"][candidateatt[0]] = candidateatt[1]
            evaluationresults = self.runClassifier(theproperty.classifier, analyzedatasets)
            fsocre = self.calculateFsocre(evaluationresults)
            if currentscore is not None:
                return fsocre - currentscore.getFMeasureValues()
            else:
                return fsocre
        except Exception as ex:
            logger.Error(f"Fone produceScore error: {ex}", ex)
            return 0

    def getEvalutionScoringMethod(self):
        return evalutionScoringMethod.Fone