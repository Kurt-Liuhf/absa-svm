import numpy as np
from numpy import double
from scipy import stats
from MAFC_Operator.operator_base import outputType
from MAFC_Operator.ColumnInfo import ColumnInfo
from MAFC_Operator.Unary.discretizer import Discretizer
from logger.logger import logger
from properties.properties import theproperty


class StatisticOperation:
    def __init__(self):
        pass

    def _calculatePairedTTestValues(self, dataset, list1, list2):
        tTestValues = []
        for ci1 in list1:
            if ci1.getType() != outputType.Numeric:
                logger.Error("Unable to process non-numeric columns - list 1")
            for ci2 in list2:
                if ci2.getType() != outputType.Numeric:
                    logger.Error("Unable to process non-numeric columns - list 2")
                if theproperty.dataframe == "dask":
                    data1 = dataset[ci1.getName()].values.compute()
                    data2 = dataset[ci2.getName()].values.compute()
                elif theproperty.dataframe == "pandas":
                    data1 = dataset[ci1.getName()].values
                    data2 = dataset[ci2.getName()].values
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
                testValue = abs(stats.ttest_rel(data1, data2))
                if testValue != None:
                    tTestValues.append(testValue)
        return tTestValues

    def calculatePairedTTestValues(self, dataset, list1, columnInfo):
        templist = []
        templist += [columnInfo]
        return self._calculatePairedTTestValues(dataset, list1, templist)

    def _calculateChiSquareTestValues(self, dataset, list1, list2):
        bins = theproperty.DiscretizerBinsNumber
        dizr = Discretizer(bins)
        chiSquareValues = []
        for ci1 in list1:
            if ci1.getType() != outputType.Discrete and ci1.getType() != outputType.Numeric:
                logger.Error("Unsupported Column Type")
            for ci2 in list2:
                if ci2.getType() != outputType.Discrete and ci2.getType() != outputType.Numeric:
                    logger.Error("Unsupported Column Type")
                # tempcolumn1 = None
                # tempcolumn2 = None
                # datalist1 = None
                # datalist2 = None
                if ci1.getType() == outputType.Numeric:
                    tempcolumn1, datalist1 = self.discretizeNumericColumn(dataset, ci1, dizr)
                else:
                    tempcolumn1 = ci1
                    datalist1 = list(dataset[tempcolumn1.getName()].values.compute())
                if ci2.getType() == outputType.Numeric:
                    tempcolumn2, datalist2 = self.discretizeNumericColumn(dataset, ci2, dizr)
                else:
                    tempcolumn2 = ci2
                    datalist2 = list(dataset[tempcolumn2.getName()].values.compute())

                templist = self.generateDiscreteAttributesCategoryIntersection(datalist1, datalist2,
                                                                               tempcolumn1.getNumsOfUnique(),
                                                                               tempcolumn2.getNumsOfUnique())

                chiSquareTestVal = self.chisquare(templist)
                if chiSquareTestVal != None:
                    chiSquareValues.append(chiSquareTestVal)
        return chiSquareValues

    def discretizeNumericColumn(self, dataset, columninfo, dizr):
        if dizr == None:
            bins = theproperty.DiscretizerBinsNumber
            dizr = Discretizer(bins)
        tempcolumnlist = []
        tempcolumnlist.append(columninfo)
        sclist = [{'name': tl.getName(), 'type': tl.getType()} for tl in tempcolumnlist]
        dizr.processTrainingSet(dataset, sclist, None)
        newcolumn = dizr.generateColumn(dataset, sclist, None)
        # datas = datadict['data'].compute()

        thecolumn = ColumnInfo([columninfo], None, dizr, dizr.getName(), False, dizr.getType(), dizr.getNumofBins())
        return thecolumn, newcolumn['data']

    def calculateChiSquareTestValues(self, dataset, list1, columnInfo):
        templist = []
        templist += [columnInfo]
        return self._calculateChiSquareTestValues(dataset, list1, templist)

    def generateDiscreteAttributesCategoryIntersection(self, data, col1: list, col2: list, n: int = theproperty.DiscretizerBinsNumber, m: int = theproperty.DiscretizerBinsNumber):
        try:
            newcol1 = col1
            newcol2 = col2
            if type(col1) != list:
                n = col1.getNumsOfUnique()
                if theproperty.dataframe == "dask":
                    newcol1 = data[col1.getName()].values.compute()
                elif theproperty.dataframe == "pandas":
                    newcol1 = data[col1.getName()].values
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
            if type(col2) != list:
                m = col2.getNumsOfUnique()
                if theproperty.dataframe == "dask":
                    newcol2 = data[col2.getName()].values.compute()
                elif theproperty.dataframe == "pandas":
                    newcol2 = data[col2.getName()].values
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
            if len(newcol1) != len(newcol2):
                return None

            list1 = np.zeros((n, m), "int32")
            for i in range(0, len(newcol1)):
                list1[newcol1[i]][newcol2[i]] += 1
        except Exception as ex:
            logger.Error(f"StatisticOperator generateDiscreteAttributesCategoryIntersection Error :{ex}", ex)
        finally:
            return list1

    def chisquare(self, list1: list):
        m = len(list1)
        n = len(list1[0])
        rowsum = np.zeros((m), "float32")
        colsum = np.zeros((n), "float32")
        total = 0.0
        for i in range(0, m):
            for j in range(0, n):
                rowsum[i] += list1[i][j]
                colsum[j] += list1[i][j]
                total += list1[i][j]
        sumsq = 0.0
        expected = 0.0
        for i in range(0, m):
            for j in range(0, n):
                if total == 0:
                    return 0
                expected = rowsum[i] * colsum[j] / total
                if expected == 0:
                    continue
                sumsq += ((float)(list1[i][j] - expected) ** 2) / expected
        return sumsq
