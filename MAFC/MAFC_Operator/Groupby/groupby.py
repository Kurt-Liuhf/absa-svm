from abc import ABC
from logger.logger import logger
from MAFC_Operator.operator_base import Operator, operatorType, outputType


class Groupby(Operator):

    def __init__(self):
        pass

    def generateName(self, sourcecolumns, targetcolumn):
        Sname = "Source("
        for sc in sourcecolumns:
            sname = sc['name']
            Sname += sname
            Sname += ";"
        Sname += ")"
        Tname = "Target("
        for tc in targetcolumn:
            tname = tc['name']
            Tname += tname
            Tname += ";"
        Tname += ")"

        name = Sname + "_" + Tname
        return name

    def getType(self) -> operatorType:
        return operatorType.GroupBy

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if targetColumns is not None and len(targetColumns) != 1:
            return False
        for sc in sourceColumns:
            if sc['type'] != outputType.Discrete:
                return False
        return True

    def processTrainingSet(self, dataset, sourceColumns: list, targetColumns: list):
        pass