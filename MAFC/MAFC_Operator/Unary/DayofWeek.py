from MAFC_Operator.Unary.unary import Unary
from MAFC_Operator.operator_base import outputType
from properties.properties import theproperty
from logger.logger import logger


class DayofWeek(Unary):
    def __init__(self):
        super(DayofWeek, self).__init__()

    def requiredInputType(self) -> outputType:
        return outputType.Date

    def processTrainingSet(self, dataset, sourceColumns, targetColumns):
        pass

    def generateColumn(self,dataset, sourceColumns, targetColumns):
        '''
        :param sourceColumns:{'name':列名,'type':outputType}
        :param targetColumns:{'name':列名,'type':outputType}
        :return: dask.dataframe.series
        '''
        columnname = str(sourceColumns[0]['name'])

        def getweek(date):
            return date.weekday()

        if theproperty.dataframe == "dask":
            columndata = dataset[columnname].apply(getweek, meta=('getweek', 'int32'))
        elif theproperty.dataframe == "pandas":
            columndata = dataset[columnname].apply(getweek)
        else:
            logger.Info(f"no {theproperty.dataframe} can use")

        name = "DayofWeek(" + columnname + ")"
        newcolumn = {"name": name, "data": columndata}
        return newcolumn

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if super(DayofWeek, self).isMatch(dataset, sourceColumns,targetColumns):
            if sourceColumns[0]['type'] == outputType.Date:
               return True
        return False

    def getNumofBins(self) -> int:
        return 7

    def getName(self):
        return "DayofWeek"

    def getOutputType(self) -> outputType:
        return outputType.Discrete
