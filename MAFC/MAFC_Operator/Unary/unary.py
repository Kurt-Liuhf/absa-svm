from abc import abstractmethod
from MAFC_Operator.operator_base import Operator, outputType
from MAFC_Operator.operator_base import operatorType


class Unary(Operator):

    def __init__(self):
        super().__init__()

    def getType(self) -> operatorType:
        return operatorType.Unary

    @abstractmethod
    def requiredInputType(self) -> outputType:
        raise NotImplementedError("Unary is an abstract class")

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:

        if len(sourceColumns) != 1 or (targetColumns is not None and len(targetColumns) != 0):
            return False
        else:
            return True

    @abstractmethod
    def getNumofBins(self) -> int:
        pass


