from MAFC_Operator.Unary.unary import Unary
from MAFC_Operator.operator_base import outputType
from properties.properties import theproperty
from logger.logger import logger


class StdOperator(Unary):
    def __init__(self):
        pass

    def requiredInputType(self) -> outputType:
        return outputType.Numeric

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def processTrainingSet(self, dataset, sourceColumns, targetColumns):

        columnname = str(sourceColumns[0]['name'])
        if theproperty.dataframe == "dask":
            self.meanval = dataset[columnname].mean().compute()
            self.stdval = dataset[columnname].std().compute()
        elif theproperty.dataframe == "pandas":
            self.meanval = dataset[columnname].mean()
            self.stdval = dataset[columnname].std()
        else:
            logger.Info(f"no {theproperty.dataframe} can use")


    def getName(self) -> str:
        return "StdOperator"

    def generateColumn(self, dataset, sourceColumns, targetColumns):
        try:
            columnname = str(sourceColumns[0]['name'])
            name = "StdOperator(" + columnname + ")"

            def getstd(data, meanval, stdval):
                if self.stdval == 0:
                    return 0.0
                return (data - meanval) / stdval

            if theproperty.dataframe == "dask":
                columndata = dataset[columnname].apply(getstd, meanval=self.meanval, stdval=self.stdval, meta=('getstd', 'float64'))
            elif theproperty.dataframe == "pandas":
                columndata = dataset[columnname].apply(getstd, meanval=self.meanval, stdval=self.stdval)
            else:
                logger.Info(f"no {theproperty.dataframe} can use")

            newcolumn = {"name": name, "data": columndata}
        except Exception as ex:
            logger.Error("Discretizer generateColumn Error}", ex)
        finally:
            return newcolumn

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if super(StdOperator, self).isMatch(dataset, sourceColumns, targetColumns):
            if sourceColumns[0]['type'] == outputType.Numeric:
                return True
        return False

    def getNumofBins(self) -> int:
        return -1