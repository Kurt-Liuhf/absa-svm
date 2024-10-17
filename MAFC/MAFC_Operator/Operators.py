from MAFC_Operator.operator_base import outputType

class Operators:

    def __init__(self, sourcecolumn, targetcolumn, operator, secondoperator):#将构造数据加入数据集.
        self.sourceColumns = sourcecolumn
        self.targetColumns = targetcolumn
        self.operator = operator
        self.secondoperators = secondoperator
        self.filterscore = 0.0
        self.wrapperscore = 0.0

    def getOperator(self):
        return self.operator

    def getType(self):
        if self.secondoperators != None:
            return self.secondoperators.getOutputType()
        elif self.operator != None:
            return self.operator.getOutputType()
        raise("Operator error!")

    def getName(self):
        str = "{sources:["
        for columnname in self.sourceColumns:
            str = str + columnname.getName()
            str = str + ","
        str = str + "];"
        str = str + "Targets:["
        if self.targetColumns is not None:
            for columnname in self.targetColumns :
                str = str + columnname.getName()
                str = str + ","
        str = str + "];"
        str = str + self.operator.getName()
        if self.secondoperators is not None:
            str = str + ","
            str = str + self.secondoperators.getName()
        str = str + "}"
        return str

    def getFScore(self):
        return self.filterscore

    def getWScore(self):
        return self.wrapperscore

    def setFScore(self,fscore):
        self.filterscore = fscore

    def setWScore(self,wscore):
        self.wrapperscore = wscore

    def __gt__(self, other):
        return self.filterscore > other.filterscore

    def __eq__(self, other):
        if other is None:
            return False
        return self.getName() == other.getName() and self.getType() == other.getType()

    def getNumofBins(self):
        if self.secondoperators is not None:
            if self.secondoperators.getOutputType() == outputType.Discrete:
                return self.secondoperators.getNumofBins()
        else:
            if self.operator.getOutputType() == outputType.Discrete:
                return self.operator.getNumofBins()
        return -1

    def __hash__(self):
        return hash(self.getName())


