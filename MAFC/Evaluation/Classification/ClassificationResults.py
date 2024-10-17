

class ClassificationResults:

    def __init__(self, auc, logloss = None, fMeasureValues = None):
        self.auc = auc
        self.logLoss = logloss
        self.fMeasureValues = fMeasureValues

    def getAuc(self):
        return self.auc

    def getLogLoss(self):
        return self.logLoss

    def getFMeasureValues(self):
        return self.fMeasureValues
