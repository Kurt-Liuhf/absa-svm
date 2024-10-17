from abc import abstractmethod
from MAFC_Operator.operator_base import Operator, outputType
from MAFC_Operator.operator_base import operatorType


class Binary(Operator):

    def __init__(self):
        super().__init__()

    def getType(self) -> operatorType:
        return operatorType.Binary

    @abstractmethod
    def requiredInputType(self) -> outputType:
        raise NotImplementedError("requiredInputType is an abstract method")

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if sourceColumns is None or targetColumns is None:
            return False
        if len(sourceColumns) != 1 or len(targetColumns) != 1 :
            return False
        if sourceColumns[0]['type'] != outputType.Numeric or targetColumns[0]['type'] != outputType.Numeric:
            return False
        return True

    def generateName(self,sourcecolumns,targetcolumn):
        name = "(" + str(sourcecolumns[0]['name']) + ";" + targetcolumn[0]['name'] + ")"
        return name

    def processTrainingSet(self, dataset, sourceColumns, targetColumns):
        pass


