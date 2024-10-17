from MAFC_Operator.Unary.unary import Unary
from MAFC_Operator.operator_base import outputType
from properties.properties import theproperty
from logger.logger import logger


class IsWeekend(Unary):
    def __init__(self):
        super(IsWeekend, self).__init__()

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

        def isweekend(date):
            if date.weekday() == 6 or date.weekday() == 5:
                return 1
            return 0

        if theproperty.dataframe == "dask":
            columndata = dataset[columnname].apply(isweekend, meta=('isweekend', 'int32'))
        elif theproperty.dataframe == "pandas":
            columndata = dataset[columnname].apply(isweekend)
        else:
            logger.Info(f"no {theproperty.dataframe} can use")

        name = "IsWeekend(" + columnname + ")"
        newcolumn = {"name": name, "data": columndata}
        return newcolumn

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if super(IsWeekend, self).isMatch(dataset,sourceColumns,targetColumns):
            if sourceColumns[0]['type'] == outputType.Date:
               return True
        return False

    def getNumofBins(self) -> int:
        return 2

    def getName(self):
        return "IsWeekend"

    def getOutputType(self) -> outputType:
        return outputType.Discrete
