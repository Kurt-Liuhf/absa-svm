from properties.properties import theproperty
import numpy as np
import dask.dataframe as dd
from scipy import stats
from MAFC_Operator.OperatorManager import OperatorManager
from Evaluation.FEvaluation.AttributeInfo import AttributeInfo
from Evaluation.FEvaluation.InformationGain import InformationGainFilterEvaluator
from Evaluation.FEvaluation.StatisticOperation import StatisticOperation
from MAFC_Operator.Operators import Operators
from logger.logger import logger
from MAFC_Operator.operator_base import outputType, operatorType


class OperatorBasedAttributes:
    def __init__(self):
        self.statisticoperation = StatisticOperation()

        self.numsOfSources: float = 0.0
        self.numOfNumericSources: float = 0.0
        self.numOfDiscreteSources: float = 0.0
        self.numOfDateSources: float = 0.0
        self.operatorTypeIdentifier: int = 0

        self.isOutputDiscrete: int = 0
        self.discreteInuse: int = 0
        self.normalizerInUse: int = 0
        self.numOfDiscreteValues: float = 0.0
        self.maxNumOfDiscreteSourceAttribtueValues: float = 0.0

        self.minNumOfDiscreteSourceAttribtueValues: float = 0.0
        self.avgNumOfDiscreteSourceAttribtueValues: float = 0.0
        self.stdNumOfDiscreteSourceAttribtueValues: float = 0.0
        self.maxValueOfNumericTargetAttribute: float = 0.0
        self.minValueOfNumericTargetAttribute: float = 0.0

        self.avgValueOfNumericTargetAttribute: float = 0.0
        self.stdValueOfNumericTargetAttribute: float = 0.0
        self.chiSquareTestValueForSourceAttributes: float = 0.0
        self.pairedTTestValueForSourceAndTargetAttirbutes: float = 0.0
        self.maxChiSquareTsetForSourceAndTargetAttributes: float = 0.0

        self.minChiSquareTsetForSourceAndTargetAttributes: float = 0.0
        self.avgChiSquareTsetForSourceAndTargetAttributes: float = 0.0
        self.stdChiSquareTsetForSourceAndTargetAttributes: float = 0.0
        self.maxChiSquareTestvalueForSourceDatasetAttributes: float = 0.0
        self.minChiSquareTestvalueForSourceDatasetAttributes: float = 0.0

        self.avgChiSquareTestvalueForSourceDatasetAttributes: float = 0.0
        self.stdChiSquareTestvalueForSourceDatasetAttributes: float = 0.0
        self.IGScore: float = 0.0

    def getOperatorsBasedAttributes(self, datadict, oa, evaluationatt):
        '''
        为所生成属性的“父级”生成元特征，这些元特征不需要计算要计算的生成属性的值
        :param evaluationatt: [name: str, data:data.dataframe.core.Series, ColumnInfo]
        :param datadict:
        :param oa: Operators
        :return: {}
        '''
        try:
            tempList = []
            tempList.append(evaluationatt)
            igfe = InformationGainFilterEvaluator()
            if evaluationatt is not None:
                igfe.initFEvaluation(tempList)
                self.IGScore = igfe.produceScore(datadict, None, None, None)

            self.ProcessOperators(oa)

            self.processSourceAndTargetAttributes(datadict["data"], oa)

            self.performStatisticalTestsOnSourceAndTargetAttributes(datadict['data'], oa)

            self.performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(datadict, oa)

        except Exception as ex:
            logger.Error(f'Failed in func "getOperatorsBasedAttributes" with exception: {ex}')
        finally:

            return self.generateInstanceAttributesMap()

    def ProcessOperators(self, oa: Operators):
        '''
        分析Operators对象的特征——构成该对象的特征。这里不处理分析的属性本身。
        :param datadict:
        :param oa:
        :return:
        '''
        if oa.sourceColumns is not None:
            self.numsOfSources = len(oa.sourceColumns)

        if oa.sourceColumns is not None:
            for ci in oa.sourceColumns:
                if ci.getType() == outputType.Numeric:
                    self.numOfNumericSources += 1
                elif ci.getType() == outputType.Discrete:
                    self.numOfDiscreteSources += 1
                elif ci.getType() == outputType.Date:
                    self.numOfDateSources += 1
        self.operatorTypeIdentifier = self.getOperatorTypeID(oa.getOperator().getType())

        if oa.secondoperators is not None:
            if oa.secondoperators.getOutputType() == outputType.Discrete:
                self.isOutputDiscrete = 1
        else:
            if oa.getOperator().getOutputType() == outputType.Discrete:
                self.isOutputDiscrete = 1
        self.discreteInuse = self.getDiscreteID(oa.secondoperators)
        self.normalizerInUse = self.getNormalizerID(oa.secondoperators)
        self.numOfDiscreteValues = self.getNumOfNewAttributeDiscreteValues(oa)

    def getOperatorTypeID(self, type):
        if type == operatorType.Unary:
            return 0
        elif type == operatorType.Binary:
            return 1
        elif type == operatorType.GroupBy:
            return 2
        elif type == operatorType.TimeGroupBy:
            return 3
        else:
            logger.Error("operatorType Error")

    def getDiscreteID(self, oper):
        if oper is None:
            return 0

        namedict = {"Discretizer": 1, "DayofWeek": 2, "HourofDay": 3, "IsWeekend": 4}
        if namedict.get(oper.getName()) != None:
            return namedict.get(oper.getName())
        return 0

    def getNormalizerID(self,oper):
        if oper is None:
            return 0
        if oper.getName() == "StdOperator":
            return 1
        return 0

    def getNumOfNewAttributeDiscreteValues(self, oa):
        if oa.secondoperators is not None:
            return oa.secondoperators.getNumOfBins()
        else:
            if oa.getOperator().getOutputType() != outputType.Discrete:
                return 0.0
            else:
                return oa.getOperator().getNumofBins()

    def processSourceAndTargetAttributes(self, dataset: dd.DataFrame, oa: Operators):
        '''

        :param oa:
        :return:
        '''
        try:
            sourceattributelist = []
            for sourceatts in oa.sourceColumns:
                if sourceatts.getType() == outputType.Discrete:
                    sourceattributelist.append(sourceatts.getNumsOfUnique())


            if len(sourceattributelist) != 0:
                self.maxNumOfDiscreteSourceAttribtueValues = np.max(sourceattributelist)
                self.minNumOfDiscreteSourceAttribtueValues = np.min(sourceattributelist)
                self.avgNumOfDiscreteSourceAttribtueValues = np.mean(sourceattributelist)
                self.stdNumOfDiscreteSourceAttribtueValues = np.std(sourceattributelist)



            if oa.targetColumns is None or len(oa.targetColumns) == 0 or oa.targetColumns[0].getType() != outputType.Numeric:
                pass
            else:
                columnname = oa.targetColumns[0].getName()
                if columnname not in dataset.columns:
                    logger.Info(columnname + " is not exist in datadict['data']")
                else:
                    if theproperty.dataframe == "dask":
                        self.maxValueOfNumericTargetAttribute = dataset[columnname].max().compute()
                        self.minValueOfNumericTargetAttribute = dataset[columnname].min().compute()
                        self.avgValueOfNumericTargetAttribute = dataset[columnname].mean().compute()
                        self.stdValueOfNumericTargetAttribute = dataset[columnname].std().compute()
                    elif theproperty.dataframe == "pandas":
                        self.maxValueOfNumericTargetAttribute = dataset[columnname].max()
                        self.minValueOfNumericTargetAttribute = dataset[columnname].min()
                        self.avgValueOfNumericTargetAttribute = dataset[columnname].mean()
                        self.stdValueOfNumericTargetAttribute = dataset[columnname].std()
                    else:
                        logger.Info(f"no {theproperty.dataframe} can use")
        except Exception as ex:
            logger.Error(f"processSourceAndTargetAttributes error: {ex}")

    def performStatisticalTestsOnSourceAndTargetAttributes(self, dataset, oa: Operators):
        '''
        离散源特征的卡方分布
        :param dataset:
        :param oa:
        :return:
        '''
        try:
            if len(oa.sourceColumns) == 2:
                if oa.sourceColumns[0].getType() == outputType.Discrete and oa.sourceColumns[1].getType() == outputType.Discrete:
                    if theproperty.dataframe == "dask":
                        templist1 = list(dataset[oa.sourceColumns[0].getName()].values.compute())
                        templist2 = list(dataset[oa.sourceColumns[1].getName()].values.compute())
                    elif theproperty.dataframe == "pandas":
                        templist1 = list(dataset[oa.sourceColumns[0].getName()].values)
                        templist2 = list(dataset[oa.sourceColumns[1].getName()].values)
                    else:
                        logger.Info(f"no {theproperty.dataframe} can use")
                    list1 = self.statisticoperation.generateDiscreteAttributesCategoryIntersection(dataset, templist1, templist2, oa.sourceColumns[0].getNumsOfUnique() , oa.sourceColumns[1].getNumsOfUnique())
                    tempval = None
                    if list1 is not None:
                        tempval = self.statisticoperation.chisquare(list1)
                    if tempval is not None:
                        self.chiSquareTestValueForSourceAttributes = tempval
                    else:
                        self.chiSquareTestValueForSourceAttributes = 0

            if len(oa.sourceColumns) == 1 and oa.sourceColumns[0].getType() == outputType.Numeric and oa.targetColumns is not None and len(oa.targetColumns) == 1:
                if theproperty.dataframe == "dask":
                    templist1 = list(dataset[oa.sourceColumns[0].getName()].values.compute())
                    templist2 = list(dataset[oa.targetColumns[0].getName()].values.compute())
                elif theproperty.dataframe == "pandas":
                    templist1 = list(dataset[oa.sourceColumns[0].getName()].values)
                    templist2 = list(dataset[oa.targetColumns[0].getName()].values)
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
                self.pairedTTestValueForSourceAndTargetAttirbutes = stats.ttest_rel(templist1, templist2)[0]

            if len(oa.sourceColumns) == 1 and oa.targetColumns == None:
                pass
            else:
                columnstoanalyze = []
                for ci in oa.sourceColumns:
                    if ci.getType() == outputType.Discrete:
                        columnstoanalyze.append(ci)
                    else:
                        if ci.getType() == outputType.Numeric:
                            newcolumn, thedata = self.statisticoperation.discretizeNumericColumn(dataset, ci, None)
                            columnstoanalyze.append(newcolumn)
                            if newcolumn.getName() not in dataset.columns:
                                dataset[newcolumn.getName()] = thedata
                if len(columnstoanalyze) > 1:
                    chiSquareTestValues = []
                    for i in range(0, len(columnstoanalyze) - 1):
                        for j in range(i+1, len(columnstoanalyze)):
                            if theproperty.dataframe == "dask":
                                templist1 = list(dataset[columnstoanalyze[i].getName()].values.compute())
                                templist2 = list(dataset[columnstoanalyze[j].getName()].values.compute())
                            elif theproperty.dataframe == "pandas":
                                templist1 = list(dataset[columnstoanalyze[i].getName()].values)
                                templist2 = list(dataset[columnstoanalyze[j].getName()].values)
                            else:
                                logger.Info(f"no {theproperty.dataframe} can use")

                            list1 = self.statisticoperation.generateDiscreteAttributesCategoryIntersection(dataset, templist1, templist2, columnstoanalyze[i].getNumsOfUnique() , columnstoanalyze[j].getNumsOfUnique())
                            chiSquareTestVal = self.statisticoperation.chisquare(list1)
                            if list1 is not None:
                                continue
                            if chiSquareTestVal != None:
                                chiSquareTestValues.append(chiSquareTestVal)
                        if len(chiSquareTestValues) > 0:
                            self.maxChiSquareTsetForSourceAndTargetAttributes = np.max(chiSquareTestValues)
                            self.minChiSquareTsetForSourceAndTargetAttributes = np.min(chiSquareTestValues)
                            self.avgChiSquareTsetForSourceAndTargetAttributes = np.mean(chiSquareTestValues)
                            self.stdChiSquareTsetForSourceAndTargetAttributes = np.std(chiSquareTestValues)
        except Exception as ex:
            logger.Error(f"performStatisticalTestsOnSourceAndTargetAttributes Error:{ex}", ex)

    def performStatisticalTestOnOperatorAssignmentAndDatasetAtributes(self, datadict, oa: Operators):
        columnstoanalyze = []
        for ci in oa.sourceColumns:
            if ci.getType() == outputType.Discrete:
                columnstoanalyze.append(ci)
        if oa.targetColumns is not None:
            for ci in oa.targetColumns:
                if ci.getType() == outputType.Discrete:
                    columnstoanalyze.append(ci)

        chisquaretestvalues = []
        dataset = datadict["data"]
        for ci in columnstoanalyze:
            for dataci in datadict["Info"]:
                if dataci in oa.sourceColumns or dataci in oa.targetColumns:
                    continue

                chisquaretestvalue = None
                if dataci.getType() == outputType.Discrete:
                    if theproperty.dataframe == "dask":
                        templist1 = list(dataset[ci.getName()].values.compute())
                        templist2 = list(dataset[dataci.getName()].values.compute())
                    elif theproperty.dataframe == "pandas":
                        templist1 = list(dataset[ci.getName()].values)
                        templist2 = list(dataset[dataci.getName()].values)
                    else:
                        logger.Info(f"no {theproperty.dataframe} can use")

                    list1 = self.statisticoperation.generateDiscreteAttributesCategoryIntersection(dataset, templist1,
                                                                        templist2, ci.getNumsOfUnique(), dataci.getNumsOfUnique())
                    if list1 is not None:
                        chisquaretestvalue = self.statisticoperation.chisquare(list1)
                if chisquaretestvalue is not None:
                    chisquaretestvalues.append(chisquaretestvalue)
                else:
                    chisquaretestvalues.append(0)



        if len(chisquaretestvalues) > 0:
            self.maxChiSquareTestvalueForSourceDatasetAttributes = np.max(chisquaretestvalues)
            self.minChiSquareTestvalueForSourceDatasetAttributes = np.min(chisquaretestvalues)
            self.avgChiSquareTestvalueForSourceDatasetAttributes = np.mean(chisquaretestvalues)
            self.stdChiSquareTestvalueForSourceDatasetAttributes = np.std(chisquaretestvalues)


    def generateInstanceAttributesMap(self):
        attributes = {}
        attributes[len(attributes)] = AttributeInfo("numsOfSources", outputType.Numeric,
                                                    self.numsOfSources, -1)
        attributes[len(attributes)] = AttributeInfo("numOfNumericSources", outputType.Numeric,
                                                    self.numOfNumericSources, -1)
        attributes[len(attributes)] = AttributeInfo("numOfDiscreteSources", outputType.Numeric,
                                                    self.numOfDiscreteSources, -1)
        attributes[len(attributes)] = AttributeInfo("numOfDateSources", outputType.Numeric,
                                                    self.numOfDateSources, -1)
        attributes[len(attributes)] = AttributeInfo("isOutputDiscrete", outputType.Discrete,
                                                    self.isOutputDiscrete, 2)


        attributes[len(attributes)] = AttributeInfo("operatorTypeIdentifier", outputType.Discrete,
                                                    self.operatorTypeIdentifier, 4)
        attributes[len(attributes)] = AttributeInfo("discreteInuse", outputType.Discrete,
                                                    self.discreteInuse, 2)
        attributes[len(attributes)] = AttributeInfo("normalizerInUse", outputType.Discrete,
                                                    self.normalizerInUse, 2)
        attributes[len(attributes)] = AttributeInfo("numOfDiscreteValues", outputType.Numeric,
                                                    self.numOfDiscreteValues, -1)
        attributes[len(attributes)] = AttributeInfo("IGScore", outputType.Numeric,
                                                    self.IGScore, -1)


        attributes[len(attributes)] = AttributeInfo("maxNumOfDiscreteSourceAttribtueValues",
                                                    outputType.Numeric,
                                                    self.maxNumOfDiscreteSourceAttribtueValues, -1)
        attributes[len(attributes)] = AttributeInfo("minNumOfDiscreteSourceAttribtueValues",
                                                    outputType.Numeric,
                                                    self.minNumOfDiscreteSourceAttribtueValues, -1)
        attributes[len(attributes)] = AttributeInfo("avgNumOfDiscreteSourceAttribtueValues",
                                                    outputType.Numeric,
                                                    self.avgNumOfDiscreteSourceAttribtueValues, -1)
        attributes[len(attributes)] = AttributeInfo("stdNumOfDiscreteSourceAttribtueValues",
                                                    outputType.Numeric,
                                                    self.stdNumOfDiscreteSourceAttribtueValues, -1)
        attributes[len(attributes)] = AttributeInfo("maxValueOfNumericTargetAttribute",
                                                    outputType.Numeric,
                                                    self.maxValueOfNumericTargetAttribute, -1)


        attributes[len(attributes)] = AttributeInfo("minValueOfNumericTargetAttribute",
                                                    outputType.Numeric,
                                                    self.minValueOfNumericTargetAttribute, -1)
        attributes[len(attributes)] = AttributeInfo("avgValueOfNumericTargetAttribute",
                                                    outputType.Numeric,
                                                    self.avgValueOfNumericTargetAttribute, -1)
        attributes[len(attributes)] = AttributeInfo("stdValueOfNumericTargetAttribute",
                                                    outputType.Numeric,
                                                    self.stdValueOfNumericTargetAttribute, -1)
        attributes[len(attributes)] = AttributeInfo("chiSquareTestValueForSourceAttributes",
                                                    outputType.Numeric,
                                                    self.chiSquareTestValueForSourceAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("pairedTTestValueForSourceAndTargetAttirbutes",
                                                    outputType.Numeric,
                                                    self.pairedTTestValueForSourceAndTargetAttirbutes, -1)


        attributes[len(attributes)] = AttributeInfo("maxChiSquareTsetForSourceAndTargetAttributes",
                                                    outputType.Numeric,
                                                    self.maxChiSquareTsetForSourceAndTargetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("minChiSquareTsetForSourceAndTargetAttributes",
                                                    outputType.Numeric,
                                                    self.minChiSquareTsetForSourceAndTargetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("avgChiSquareTsetForSourceAndTargetAttributes",
                                                    outputType.Numeric,
                                                    self.avgChiSquareTsetForSourceAndTargetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("stdChiSquareTsetForSourceAndTargetAttributes",
                                                    outputType.Numeric,
                                                    self.stdChiSquareTsetForSourceAndTargetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("maxChiSquareTestvalueForSourceDatasetAttributes",
                                                    outputType.Numeric,
                                                    self.maxChiSquareTestvalueForSourceDatasetAttributes, -1)


        attributes[len(attributes)] = AttributeInfo("minChiSquareTestvalueForSourceDatasetAttributes",
                                                    outputType.Numeric,
                                                    self.minChiSquareTestvalueForSourceDatasetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("avgChiSquareTestvalueForSourceDatasetAttributes",
                                                    outputType.Numeric,
                                                    self.avgChiSquareTestvalueForSourceDatasetAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("stdChiSquareTestvalueForSourceDatasetAttributes",
                                                    outputType.Numeric,
                                                    self.stdChiSquareTestvalueForSourceDatasetAttributes, -1)

        return attributes

