

class ClassificationItem:

    def __init__(self, trueClass, probabilities):
        self.trueClass = trueClass
        self.probabilities = probabilities

    def getTrueClass(self):
        return self.trueClass

    def getProbabilities(self):
        return self.probabilities

    def setProbabilities(self, probabilities):
        self.probabilities = probabilities

    def setProbabilityOfClass(self, classIdx, value):
        self.probabilities[classIdx] = value

    def getProbabilitiesOfClass(self,  index):
        return self.probabilities[index]