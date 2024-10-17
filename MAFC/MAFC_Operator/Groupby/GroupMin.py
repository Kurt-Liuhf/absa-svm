from MAFC_Operator.Groupby.groupby import Groupby
from MAFC_Operator.operator_base import outputType
from logger.logger import logger
from properties.properties import theproperty


class GroupMin(Groupby):
    def __init__(self):
        self.data = {}

    def requiredInputType(self) -> outputType:
        return outputType.Discrete

    def getOutputType(self) -> outputType:
        return outputType.Numeric

    def getName(self) -> str:
        return "GroupMin"

    def processTrainingSet(self, dataset, sourceColumns: list, targetColumns: list):
        sname = []
        for sc in sourceColumns:
            sname.append(sc['name'])
        tname = targetColumns[0]['name']
        columndata = dataset.groupby(sname)[tname].agg("min")
        if theproperty.dataframe == "dask":
            thedata = columndata.compute()
        elif theproperty.dataframe == "pandas":
            thedata = columndata
        else:
            logger.Info(f"no {theproperty.dataframe} can use")
        value = [i for i in thedata.values]
        key = [tuple([i]) if type(i) == int else tuple(i) for i in thedata.index]
        if len(value) != len(key):
            logger.Info(f"GroupMin: len(value): {len(value)} != len(key) :{len(key)}")
        self.data = {}
        for i in range(0, len(value)):
            self.data[key[i]] = value[i]

    def generateColumn(self, dataset, sourceColumns, targetColumns):
        try:
            def getmin(df, sourcecolumns, datadict):
                sname = [sc['name'] for sc in sourcecolumns]
                data = df[sname]
                key = [(int)(val) for val in data]
                key = tuple(key)
                if datadict.get(key) is None:
                    logger.Info("GroupMein: self.data is not init")
                    return 0
                return datadict[key]

            columndata = None
            if theproperty.dataframe == "dask":
                columndata = dataset.apply(getmin, sourcecolumns=sourceColumns, datadict=self.data, meta=('getmin', 'float'), axis=1)
            elif theproperty.dataframe == "pandas":
                columndata = dataset.apply(getmin, sourcecolumns=sourceColumns, datadict=self.data, axis=1)
            else:
                logger.Info(f"no {theproperty.dataframe} can use")
        except Exception as ex:
            logger.Error(f"GroupMin generateColumn Error : {ex}", ex)
        finally:
            name = self.getName() + "(" + self.generateName(sourceColumns, targetColumns) + ")"
            newcolumn = {"name": name, "data": columndata}
            return newcolumn

    def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
        if super(GroupMin, self).isMatch(dataset, sourceColumns, targetColumns):
            if targetColumns[0]['type'] == outputType.Numeric:
                return True
        return False