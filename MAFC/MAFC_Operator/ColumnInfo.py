

class ColumnInfo:
    def __init__(self,sourcecolumns, targetcolumns, operator, name, istargetclass, type, numsofunique = None):
        self.sourcecolumns = sourcecolumns
        self.targetcolumns = targetcolumns
        self.operator = operator
        self.name = name
        self.istargetclass = istargetclass
        self.type = type
        self.numsofunique = numsofunique

    def getNumsOfUnique(self):
        return self.numsofunique

    def getName(self):
        return self.name

    def setSourceColumns(self, sourcecolumns):
        self.sourcecolumns = sourcecolumns

    def getSourceColumns(self):
        return self.sourcecolumns

    def setOperator(self, operator):
        self.operator = operator

    def getOperator(self):
        return self.operator

    def setTargetColumns(self, targetcolumns):
        self.targetcolumns = targetcolumns

    def getTargetColumns(self):
        return self.targetcolumns

    def setIstargetClass(self, istargetclass):
        self.istargetclass = istargetclass

    def getIstargetClass(self):
        return self.istargetclass

    def __eq__(self, other):
        if other is None:
            return False
        if self.name != other.getName() or self.type != other.getType():
            return False
        return True

    def __hash__(self):
        return hash(self.name)

    def getType(self):
        return self.type