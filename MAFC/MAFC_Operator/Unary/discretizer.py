from MAFC_Operator.Unary.unary import Unary
from MAFC_Operator.operator_base import outputType
from logger.logger import logger
from properties.properties import theproperty

class Discretizer(Unary):
    def __init__(self, upperbound=theproperty.DiscretizerBinsNumber):
        self.upperbound = upperbound

    def requiredInputType(self) -> outputType:
        return outputType.Numeric

    def getOutputType(self) -> outputType:
        return outputType.Discrete

    def processTrainingSet(self, dataset, sourceColumns, targetColumns):

        columnname = str(sourceColumns[0]['name'])
        if theproperty.dataframe == "dask":
            self.max_float = dataset[columnname].max().compute()
            self.min_float = dataset[columnname].min().compute()
        elif theproperty.dataframe == "pandas":
            self.max_float = dataset[columnname].max()
            self.min_float = dataset[columnname].min()
        else:
            logger.Info(f"no {theproperty.dataframe} can use")
        diff = self.max_float - self.min_float
        #print(diff)
        self.therange = diff / self.upperbound


    def getName(self) -> str:
        return "Discretizer"

    def generateColumn(self, dataset, sourceColumns, targetColumns):
        try:
            columnname = str(sourceColumns[0]['name'])
            def getdiscretizer(data, therange, min_float, nums):
                if therange == 0.0:
                    return 0
                try:
                    res = int((data - min_float) / therange)
                    return min(res, nums - 1)
                except Exception as ex:
                    #logger.Info(" data =  " + str(data) + " " + str(therange) + " " + str(min_float))
                    return 0
            if theproperty.dataframe == "dask":
                columndata = dataset[columnname].apply(getdiscretizer, therange=self.therange, min_float=self.min_float, nums=self.upperbound, meta=('getdiscretizer', 'int32'))
            elif theproperty.dataframe == "pandas":
                columndata = dataset[columnname].apply(getdiscretizer, therange=self.therange, min_float=self.min_float, nums=self.upperbound)
            else:
                logger.Info(f"no {theproperty.dataframe} can use")
            name = "Discretizer(" + columnname + ")"
            newcolumn = {"name": name, "data": columndata}
        except Exception as ex:
            logger.Error(f"Discretizer generateColumn Error {ex}", ex)
        finally:
            return newcolumn

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if super(Discretizer, self).isMatch(dataset, sourceColumns, targetColumns):
            if sourceColumns[0]['type'] == outputType.Numeric:
                return True
        return False

    def getNumofBins(self) -> int:
        return self.upperbound