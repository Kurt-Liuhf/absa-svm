import copy
import numpy
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from Evaluation.FEvaluation.AttributeInfo import AttributeInfo
from Evaluation.WEvaluation.AucWrapperEvaluation import AucWrapperEvaluation
from MAFC_Operator.ColumnInfo import ColumnInfo
from MAFC_Operator.operator_base import outputType
from MAFC_Operator.Unary.discretizer import Discretizer
from Evaluation.FEvaluation.InformationGain import InformationGainFilterEvaluator
from logger.logger import logger
from mytime import getNowTime
from properties.properties import theproperty
from Evaluation.FEvaluation.StatisticOperation import StatisticOperation

class DatasetAttributes:

    def __init__(self):
        #   数据集的一些基本特征
        self.numOfInstances: float = 0.0
        self.numOfFeatures: float = 0.0
        self.numOfNumericAttributes: float = 0.0
        self.numOfDiscreteAttributes: float = 0.0
        self.ratioOfNumericAttributes: float = 0.0
        self.ratioOfDiscreteAttributes: float = 0.0
        

        # 离散特征
        self.maxNumberOfDiscreteValuesPerAttribute: float = 0.0
        self.minNumberOfDiscreteValuesPerAttribute: float = 0.0
        self.avgNumOfDiscreteValuesPerAttribute: float = 0.0
        self.stdNumOfDiscreteValuesPerAttribute: float = 0.0

        # 数据集的统计学特征
        self.FMeasureValues: float = 0.0
        self.logLossValues: float = 0.0
        self.aucValues: float = 0.0

        # 交叉熵信息
        self.maxIGVal: float = 0.0
        self.minIGVal: float = 0.0
        self.avgIGVal: float = 0.0
        self.stdIGVal: float = 0.0

        self.discreteAttsMaxIGVal: float = 0.0
        self.discreteAttsMinIGVal: float = 0.0
        self.discreteAttsAvgIGVal: float = 0.0
        self.discreteAttsStdIGVal: float = 0.0

        self.numericAttsMaxIGVal: float = 0.0
        self.numericAttsMinIGVal: float = 0.0
        self.numericAttsAvgIGVal: float = 0.0
        self.numericAttsStdIGVal: float = 0.0

        # 相关性信息
        self.maxPairedTTestValueForNumericAttributes: float = 0.0
        self.minPairedTTestValueForNumericAttributes: float = 0.0
        self.avgPairedTTestValueForNumericAttributes: float = 0.0
        self.stdPairedTTestValueForNumericAttributes: float = 0.0

        self.maxChiSquareValueForDiscreteAttributes: float = 0.0
        self.minChiSquareValueForDiscreteAttributes: float = 0.0
        self.avgChiSquareValueForDiscreteAttributes: float = 0.0
        self.stdChiSquareValueForDiscreteAttributes: float = 0.0

        self.maxChiSquareValueForDiscreteAndDiscreteAttributes: float = 0.0
        self.minChiSquareValueForDiscreteAndDiscreteAttributes: float = 0.0
        self.avgChiSquareValueForDiscreteAndDiscreteAttributes: float = 0.0
        self.stdChiSquareValueForDiscreteAndDiscreteAttributes: float = 0.0

        #
        self.discreteAttributesList: list = []
        self.numericAttributesList: list = []

        self.stcop = StatisticOperation()
    def getDatasetBasedFeature(self, datadict, classifier):
        '''
        获取数据集的基本特征
        :param datadict:
        :param classifier:
        :return:dict
        '''
        try:
            #logger.Info("getDatasetBasedFeature " + getNowTime())
            self.processGeneralDatasetInfo(datadict)
            #logger.Info("processGeneralDatasetInfo " + getNowTime())
            self.processInitialEvaluationInformation(datadict, classifier)
            #logger.Info("processInitialEvaluationInformation " + getNowTime())
            self.processEntropyBasedMeasures(datadict)
            #logger.Info("processEntropyBasedMeasures " + getNowTime())
            self.processAttributesStatisticalTests(datadict)
            #logger.Info("processAttributesStatisticalTests " + getNowTime())
        except Exception as ex:
            logger.Error(f'Failed in func "getDatasetBasedFeature" with exception: {ex}')
            #return None
        finally:
            return self.generateDatasetAttributesMap()

    def processGeneralDatasetInfo(self, datadict):
        '''
        提取基本信息
        :param datadict:
        :return:
        '''
        self.numOfInstances = len(datadict["data"])
        self.numOfFeatures = len(datadict["data"].columns)

        for clin in datadict["Info"]:
            if clin.getType() == outputType.Numeric:
                self.numOfNumericAttributes += 1
                self.numericAttributesList.append(clin)

            elif clin.getType() == outputType.Discrete:
                self.numOfDiscreteAttributes += 1
                self.discreteAttributesList.append(clin)

        self.ratioOfNumericAttributes = self.numOfNumericAttributes / (
                self.numOfNumericAttributes + self.numOfDiscreteAttributes)
        self.ratioOfDiscreteAttributes = self.numOfDiscreteAttributes / (
                self.numOfNumericAttributes + self.numOfDiscreteAttributes)

        numOfValuesPerDiscreteAttribute = []
        for columnInfo in self.discreteAttributesList:
            numOfValuesPerDiscreteAttribute.append(columnInfo.getNumsOfUnique())

        if len(numOfValuesPerDiscreteAttribute) > 0:
            self.maxNumberOfDiscreteValuesPerAttribute = max(numOfValuesPerDiscreteAttribute)
            self.minNumberOfDiscreteValuesPerAttribute = min(numOfValuesPerDiscreteAttribute)
            self.avgNumOfDiscreteValuesPerAttribute = sum(numOfValuesPerDiscreteAttribute) / len(
                numOfValuesPerDiscreteAttribute)

            self.stdNumOfDiscreteValuesPerAttribute = np.asarray(numOfValuesPerDiscreteAttribute,
                                                                 dtype=np.float32).std()


    def processInitialEvaluationInformation(self, datadict, classifier: str):
        '''
        用于在初始数据字典上获取有关分类器性能的信息。对于培训数据集，需要提供整个datadict。对于测试数据集，只有训练数据集。
        :param datadict:
        :param classifier:
        :return:
        '''
        wrapperEvaluator = None
        wrapperName = 'AucWrapperEvaluator'
        if wrapperName == 'AucWrapperEvaluator':
            wrapperEvaluator = AucWrapperEvaluation()
        else:
            logger.Error('Wrapper Error')

        classificationResult = wrapperEvaluator.ProduceClassifications(datadict, classifier)

        self.aucValues = classificationResult.getAuc()
        self.logLossValues = classificationResult.getLogLoss()
        self.FMeasureValues = classificationResult.getFMeasureValues()

    def processEntropyBasedMeasures(self, datadict):
        '''
        计算交叉熵
        :param datadict:
        :return:
        '''
        IGScoresPerColumnIndex = []
        IGScoresPerDiscreteColumnIndex = []
        IGScoresPerNumericColumnIndex = []

        ige = InformationGainFilterEvaluator()
        data = datadict["data"]
        for idx in data.columns:
            ci = data[idx]

            if ci.dtype != "float64" and ci.dtype != "float32" and ci.dtype != "int64" and ci.dtype != "int32":
                continue

            indicedList = []
            indicedList.append(idx)
            # 此处留坑
            replicatedDataset = copy.deepcopy(datadict)
            column = None
            for cl in replicatedDataset["Info"]:
                if cl.getName() == ci.name:
                    column = cl
                    break
            if column is None:
                logger.Info("ci.name" + "columninfo is not exist")
            newcolumn = [ci.name, ci, column]
            tempList = []
            if newcolumn is not None:
                tempList.append(newcolumn)
                ige.initFEvaluation(tempList)
                score = ige.produceScore(replicatedDataset, None, None, None)
                IGScoresPerColumnIndex.append(score)

            if ci.dtype == "int64" or ci.dtype == "int32":
                IGScoresPerDiscreteColumnIndex.append(score)
            else:
                IGScoresPerNumericColumnIndex.append(score)

        IGScoresPerDiscreteColumnIndex = np.array(IGScoresPerDiscreteColumnIndex)
        IGScoresPerNumericColumnIndex = np.array(IGScoresPerNumericColumnIndex)
        IGScoresPerColumnIndex = np.array(IGScoresPerColumnIndex)
        self.maxIGVal = np.max(IGScoresPerColumnIndex)
        self.minIGVal = np.min(IGScoresPerColumnIndex)
        self.avgIGVal = np.mean(IGScoresPerColumnIndex)
        self.stdIGVal = np.std(IGScoresPerColumnIndex)

        if IGScoresPerDiscreteColumnIndex.shape[0] > 0:
            self.discreteAttsMaxIGVal = np.max(IGScoresPerDiscreteColumnIndex)
            self.discreteAttsMinIGVal = np.min(IGScoresPerDiscreteColumnIndex)
            self.discreteAttsAvgIGVal = np.mean(IGScoresPerDiscreteColumnIndex)
            self.discreteAttsStdIGVal = np.std(IGScoresPerDiscreteColumnIndex)


        if IGScoresPerNumericColumnIndex.shape[0] > 0:
            self.numericAttsMaxIGVal = np.max(IGScoresPerNumericColumnIndex)
            self.numericAttsMinIGVal = np.min(IGScoresPerNumericColumnIndex)
            self.numericAttsAvgIGVal = np.mean(IGScoresPerNumericColumnIndex)
            self.numericAttsStdIGVal = np.std(IGScoresPerNumericColumnIndex)


    def processAttributesStatisticalTests(self, datadict):
        '''
        用于计算数据集中不同属性的相关性。对于数值属性，我们在每一对之间进行配对T检验。对于离散属性，我们进行卡方检验。最后，我们将数值属性离散化，并对所有属性进行额外的Chi-Suqare测试。
        :param datadict:
        :return:
        '''
        try:
            pairedTTestValuesList = []
            for i in range(len(self.numericAttributesList) - 1):
                for j in range(i + 1, len(self.numericAttributesList)):
                    if i != j:
                        if theproperty.dataframe == "dask":
                            listi = datadict['data'][self.numericAttributesList[i].getName()].values.compute()
                            listj = datadict['data'][self.numericAttributesList[j].getName()].values.compute()
                        elif theproperty.dataframe == "pandas":
                            listi = np.array(datadict['data'][self.numericAttributesList[i].getName()].values)
                            listj = np.array(datadict['data'][self.numericAttributesList[j].getName()].values)
                        else:
                            logger.Info(f"no {theproperty.dataframe} can use")
                        tstat, pval = scipy.stats.ttest_ind(listi, listj)

                        tTestVal = abs(tstat)
                        if not np.isnan(tTestVal) and not np.isinf(tTestVal):
                            pairedTTestValuesList.append(tTestVal)

            if len(pairedTTestValuesList) > 0:
                self.maxPairedTTestValueForNumericAttributes = max(pairedTTestValuesList)
                self.minPairedTTestValueForNumericAttributes = min(pairedTTestValuesList)
                self.avgPairedTTestValueForNumericAttributes = np.average(pairedTTestValuesList)
                self.stdPairedTTestValueForNumericAttributes = np.std(pairedTTestValuesList)


            chiSquaredTestValuesList = []
            for i in range(len(self.discreteAttributesList) - 1):
                for j in range(i + 1, len(self.discreteAttributesList)):
                    if i != j:
                        #logger.Info(f"generateAtt Start :{getNowTime()}")
                        counts = self.generateDiscreteAttributesCategoryIntersection(datadict['data'], self.discreteAttributesList[i], self.discreteAttributesList[j])
                        #logger.Info(f"generateAtt End :{getNowTime()}")
                        if counts is None:
                            continue
                        #testVal = scipy.stats.chi2_contingency(counts)[0]
                        #logger.Info(f"chisquare Start :{getNowTime()}")
                        testVal = self.stcop.chisquare(counts)
                        #logger.Info(f"chisquare End :{getNowTime()}")
                        if not np.isnan(testVal) and not np.isinf(testVal):
                            chiSquaredTestValuesList.append(testVal)

            if len(chiSquaredTestValuesList) > 0:
                self.maxChiSquareValueForDiscreteAttributes = np.max(chiSquaredTestValuesList)
                self.minChiSquareValueForDiscreteAttributes = np.min(chiSquaredTestValuesList)
                self.avgChiSquareValueForDiscreteAttributes = np.average(chiSquaredTestValuesList)
                self.stdChiSquareValueForDiscreteAttributes = np.std(chiSquaredTestValuesList)

            # 最后，我们将数字特征离散化，并进行额外的卡方评估
            bins = theproperty.DiscretizerBinsNumber
            discretizedColumns = []
            for ci in self.numericAttributesList:
                #logger.Info(f"Discretizer Start :{getNowTime()}")
                erduo = Discretizer(bins)
                tempColumnsList = []
                tempColumnsList.append(ci)
                sourceColumnslist = [{'name': tl.getName(), 'type': tl.getType()} for tl in tempColumnsList]
                erduo.processTrainingSet(datadict['data'], sourceColumnslist, None)
                discretizedAttribute = erduo.generateColumn(datadict['data'], sourceColumnslist, None)

                if theproperty.dataframe == "dask":
                    discretizedAttdata = np.array(discretizedAttribute['data'].values.compute())

                elif theproperty.dataframe == "pandas":
                    discretizedAttdata = np.array(discretizedAttribute['data'].values)
                    #values = np.array(discretizedAttdata)

                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
                lenval = len(np.unique(discretizedAttdata))
                if lenval > 1:
                    discretizedColumns.append(discretizedAttdata)
                #logger.Info(f"Discretizer End :{getNowTime()}")

            # 现在，我们将所有原始离散属性添加到此列表中，并再次运行卡方检验
            discretizedColumns += self.discreteAttributesList
            chiSquaredTestValuesList = []
            for i in range(len(discretizedColumns) - 1):
                for j in range(i + 1, len(discretizedColumns)):
                    if i != j:
                        counts = self.generateDiscreteAttributesCategoryIntersection(datadict['data'], discretizedColumns[i], discretizedColumns[j])
                        if counts is None:
                            continue
                        testVal = self.stcop.chisquare(counts)
                        #testVal = scipy.stats.chi2_contingency(counts)[0]
                        if not np.isnan(testVal) and not np.isinf(testVal):
                            chiSquaredTestValuesList.append(testVal)

            if len(chiSquaredTestValuesList) > 0:
                self.maxChiSquareValueForDiscreteAndDiscreteAttributes = np.max(chiSquaredTestValuesList)
                self.minChiSquareValueForDiscreteAndDiscreteAttributes = np.min(chiSquaredTestValuesList)
                self.avgChiSquareValueForDiscreteAndDiscreteAttributes = np.average(chiSquaredTestValuesList)
                self.stdChiSquareValueForDiscreteAndDiscreteAttributes = np.std(chiSquaredTestValuesList)

        except Exception as ex:
            logger.Error(f"processAttributesStatisticalTests error: {ex}", ex)


    def generateDatasetAttributesMap(self):
        '''
        生成字典
        :return: dict
        '''
        attributes = {}

        attributes[len(attributes)] = AttributeInfo("numOfInstances", outputType.Numeric, self.numOfInstances, -1)
        attributes[len(attributes)] = AttributeInfo("maxPairedTTestValueForNumericAttributes", outputType.Numeric,
                              self.maxPairedTTestValueForNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("numOfFeatures", outputType.Numeric, self.numOfFeatures, -1)
        attributes[len(attributes)] = AttributeInfo("numOfNumericAttributes", outputType.Numeric, self.numOfNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("numOfDiscreteAttributes", outputType.Numeric, self.numOfDiscreteAttributes, -1)


        attributes[len(attributes)] = AttributeInfo("ratioOfNumericAttributes", outputType.Numeric, self.ratioOfNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("ratioOfDiscreteAttributes", outputType.Numeric, self.ratioOfDiscreteAttributes,-1)
        attributes[len(attributes)] = AttributeInfo("maxNumberOfDiscreteValuesPerAttribute", outputType.Numeric,
                             self.maxNumberOfDiscreteValuesPerAttribute, -1)
        attributes[len(attributes)] = AttributeInfo("minNumberOfDiscreteValuesPerAttribute", outputType.Numeric,
                             self.minNumberOfDiscreteValuesPerAttribute, -1)
        attributes[len(attributes)] = AttributeInfo("avgNumOfDiscreteValuesPerAttribute", outputType.Numeric,
                              self.avgNumOfDiscreteValuesPerAttribute, -1)


        attributes[len(attributes)] = AttributeInfo("minPairedTTestValueForNumericAttributes", outputType.Numeric,
                              self.minPairedTTestValueForNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("avgPairedTTestValueForNumericAttributes", outputType.Numeric,
                              self.avgPairedTTestValueForNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("stdPairedTTestValueForNumericAttributes", outputType.Numeric,
                              self.stdPairedTTestValueForNumericAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("maxChiSquareValueForDiscreteAttributes", outputType.Numeric,
                              self.maxChiSquareValueForDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("minChiSquareValueForDiscreteAttributes", outputType.Numeric,
                              self.minChiSquareValueForDiscreteAttributes, -1)


        attributes[len(attributes)] = AttributeInfo("avgChiSquareValueForDiscreteAttributes", outputType.Numeric,
                              self.avgChiSquareValueForDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("stdChiSquareValueForDiscreteAttributes", outputType.Numeric,
                              self.stdChiSquareValueForDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("aucValues", outputType.Numeric, self.aucValues, -1)
        attributes[len(attributes)] = AttributeInfo("maxChiSquareValueForDiscreteAndDiscreteAttributes", outputType.Numeric,
                              self.maxChiSquareValueForDiscreteAndDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("logLossValues", outputType.Numeric, self.logLossValues, -1)


        attributes[len(attributes)] = AttributeInfo("maxIGVal", outputType.Numeric, self.maxIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("minIGVal", outputType.Numeric, self.minIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("avgIGVal", outputType.Numeric, self.avgIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("stdIGVal", outputType.Numeric, self.stdIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("discreteAttsMaxIGVal", outputType.Numeric, self.discreteAttsMaxIGVal, -1)


        attributes[len(attributes)] = AttributeInfo("discreteAttsMinIGVal", outputType.Numeric, self.discreteAttsMinIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("discreteAttsAvgIGVal", outputType.Numeric, self.discreteAttsAvgIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("discreteAttsStdIGVal", outputType.Numeric, self.discreteAttsStdIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("numericAttsMaxIGVal", outputType.Numeric, self.numericAttsMaxIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("numericAttsMinIGVal", outputType.Numeric, self.numericAttsMinIGVal, -1)


        attributes[len(attributes)] = AttributeInfo("numericAttsAvgIGVal", outputType.Numeric, self.numericAttsAvgIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("numericAttsStdIGVal", outputType.Numeric, self.numericAttsStdIGVal, -1)
        attributes[len(attributes)] = AttributeInfo("minChiSquareValueForDiscreteAndDiscreteAttributes", outputType.Numeric,
                              self.minChiSquareValueForDiscreteAndDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("avgChiSquareValueForDiscreteAndDiscreteAttributes", outputType.Numeric,
                              self.avgChiSquareValueForDiscreteAndDiscreteAttributes, -1)
        attributes[len(attributes)] = AttributeInfo("stdChiSquareValueForDiscreteAndDiscreteAttributes", outputType.Numeric,
                              self.stdChiSquareValueForDiscreteAndDiscreteAttributes, -1)
        

        attributes[len(attributes)] = AttributeInfo("FMeasureValues", outputType.Numeric,
                                            self.FMeasureValues, -1)

        return attributes

    def generateDiscreteAttributesCategoryIntersection(self, data, col1, col2, n: int = theproperty.DiscretizerBinsNumber, m: int = theproperty.DiscretizerBinsNumber):
        newcol1 = col1
        newcol2 = col2
        if type(col1) == ColumnInfo:
            n = col1.getNumsOfUnique()
            if theproperty.dataframe == "dask":
                newcol1 = data[col1.getName()].values.compute()
            elif theproperty.dataframe == "pandas":
                newcol1 = data[col1.getName()].values
            else:
                logger.Info(f"no {theproperty.dataframe} can use")
        if type(col2) == ColumnInfo:
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
        return list1

