from abc import abstractmethod
from enum import Enum
import numpy as np

class operatorType(Enum):
    Unary = 0
    Binary = 1
    GroupBy = 2
    TimeGroupBy = 3

class outputType(Enum):
    Numeric = 0
    Discrete = 1
    Date = 2
    String = 3

class Operator(object):

    @staticmethod
    def getColumnType(column ) -> outputType:
        if np.issubdtype(column.dtype,np.int64):
            return outputType.Discrete
        elif np.issubdtype(column.dtype,np.float64):
            return outputType.Numeric
        elif np.issubdtype(column.dtype,np.datetime64):
            return outputType.Date
        else:
            raise Exception('Unknown column type')

    def __init__(self):
        pass

    @abstractmethod
    def getName(self) -> str:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    @abstractmethod
    def getType(self) -> operatorType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    @abstractmethod
    def getOutputType(self) -> outputType:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    @abstractmethod
    def processTrainingSet(self, dataset, sourceColumns, targetColumns):
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    @abstractmethod
    def generateColumn(self, dataset, sourceColumns, targetColumns) :
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")

    @abstractmethod
    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        raise NotImplementedError("Abstract class Operator shouldn't instanced directly")
