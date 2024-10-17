import datetime
import os.path
import copy
import threading
from shutil import copyfile
import numpy as np
import pandas as pd
from distributed import Client
from parallel.parallel import MutilProcess ,MyThreadPool
from Evaluation.FEvaluation.OperatorBasedAttributes import OperatorBasedAttributes
import dask.dataframe
from Evaluation.FEvaluation.DatasetAttributes import DatasetAttributes
from Evaluation.FEvaluation.AttributeInfo import AttributeInfo
from Evaluation.WEvaluation.WEvaluation import WEvaluation
from Serialize import serialize, deserialize
from MAFC_Operator.OperatorManager import OperatorManager
from dask_ml.wrappers import ParallelPostFit
from MAFC_Operator.operator_base import outputType
from logger.logger import logger
from properties.properties import theproperty

class MLAttributeManager:
    def __init__(self):
        pass

    def getBackgroundClassificationModel(self, datadict):
        backgroundFilePath = theproperty.rootpath + theproperty.backmodelpath + datadict["data"].name + "_model_classifier_obj"
        if os.path.isfile(backgroundFilePath):
            logger.Info(datadict["data"].name + " model has exist")
            model = deserialize(backgroundFilePath)
            return model
        else:
            datasetfilepath = theproperty.rootpath + theproperty.datasetlocation
            if os.path.isdir(datasetfilepath):
                addhead = True
                for fp in os.listdir(datasetfilepath):
                    if os.path.isfile(datasetfilepath + fp) and datadict["data"].name not in os.path.basename(datasetfilepath):
                        self.addFiletoTargetfile(backgroundFilePath, datasetfilepath + fp, addhead)
                        addhead = False
                    else:
                        logger.Info("skip file : " + fp)
            else:
                logger.Error("datasetlocation is not exist")
            if os.path.isfile(backgroundFilePath + ".csv"):
                if theproperty.dataframe == "dask":
                    datatrain = dask.dataframe.read_csv(backgroundFilePath + ".csv")
                elif theproperty.dataframe == "pandas":
                    datatrain = pd.read_csv(backgroundFilePath + ".csv")
                else:
                    logger.Info(f"no {theproperty.dataframe} can use")
                y = datatrain.iloc[:, -1]
                del datatrain[y.name]
                model = self.buildModel(datatrain, y, theproperty.classifier)
                serialize(backgroundFilePath, model)
                return model
            else:
                logger.Error("modeldata is not exist")

    def buildModel(self, datatarin, target, classifier):
        wevaluation = WEvaluation()
        model = wevaluation.getClassifier(classifier)
        if theproperty.dataframe == "dask":
            client = Client()
            clf = ParallelPostFit(model, scoring="r2")
            clf.fit(datatarin, target)
            client.close()
        elif theproperty.dataframe == "pandas":
            model.fit(datatarin, target)
            return model
        else:
            logger.Info(f"no {theproperty.dataframe} can use")
        return clf

    def addFiletoTargetfile(self, modelfilepath, datasetfilepath, addhead):
        modelfile = modelfilepath + ".csv"
        if os.path.isfile(modelfile) and not addhead:
            with open(modelfile, "a") as tarfile:
                with open(datasetfilepath, "r") as sourfile:
                    sourfile.readline()
                    a = sourfile.readline()
                    tarfile.writelines("\n")
                    while a != "":
                        #print(a)
                        tarfile.writelines(a)
                        a = sourfile.readline()
        else:
            copyfile(datasetfilepath, modelfile)



    def getDatasetInstances(self, datadict):
        filename = datadict["data"].name + "_candidatedata.csv"
        logger.Info(f"start {filename} getDatasetInstances")
        filepath = theproperty.rootpath + theproperty.datasetlocation  + filename
        if os.path.isfile(filepath):
            logger.Info(datadict["data"].name + "candidatedata has existed")
            if theproperty.dataframe == "dask":
                datainstances = dask.dataframe.read_csv(filepath)
            elif theproperty.dataframe == "pandas":
                datainstances = pd.read_csv(filepath)
            return datainstances

        dataattsvalues = self.generateTrainsetAtts(datadict)
        datainstances = self.generateValuesTabular(dataattsvalues)
        datainstances.to_csv(filepath, index=False)
        return datainstances

    def generateValuesTabularFromFE(self, candidateAttributes):
        tempattlist = [candidateAttributes]
        return self.generateValuesTabular(tempattlist)

    def generateTrainsetAtts(self, datadict):
        '''
        :param datadict:
        :return:[{}]
        '''
        try:
            candidateattslist = []
            trainsetattspath = theproperty.rootpath + theproperty.resultfilepath + datadict["data"].name + "_candidateattslist"
            if os.path.isfile(trainsetattspath):
                return deserialize(trainsetattspath)

            classifiers = theproperty.classifiersforMLAttributes
            dbas = DatasetAttributes()
            evaluator = WEvaluation()
            for classifier in classifiers:
                evaluationresult = evaluator.runClassifier(classifier, datadict)
                originalAUC = evaluator.calculateAUC(evaluationresult)

                datasetatts = dbas.getDatasetBasedFeature(datadict, classifier)

                classifieratt = AttributeInfo("Classifier", outputType.Discrete, self.getClassifierIndex(classifier), 3)

                datasetatts[len(datasetatts)] = classifieratt

                oms = OperatorManager()
                unaryoperlist = theproperty.unaryoperator
                unaryoperators = oms.UnaryOperator(datadict, unaryoperlist)

                otheroperlist = theproperty.otheroperator
                otheroperators = oms.OtherOperator(datadict, otheroperlist)
                otheroperators = unaryoperators + otheroperators
                numofthread = theproperty.thread
                index = 1
                lock = threading.Lock()

                def myfunction(ops, **kwargs):
                    try:
                        oms = OperatorManager()
                        datacopy = copy.deepcopy(kwargs['datadict'])
                        candidateatt = oms.generateColumn(datacopy["data"], ops, False)
                        if candidateatt is None:
                            return
                        obas = OperatorBasedAttributes()
                        oms.addColumn(datacopy, candidateatt)
                        candidateattsdict = obas.getOperatorsBasedAttributes(datacopy, ops, candidateatt)

                        evaluationresult = kwargs['evaluator'].runClassifier(kwargs['classifier'], datacopy)
                        auc = kwargs['evaluator'].calculateAUC(evaluationresult)
                        deltaAUC = auc - kwargs['originalAUC']
                        if deltaAUC > 0.01:
                            classatt = AttributeInfo("classattribute", outputType.Discrete, 1, 2)
                            logger.Info("find positive match")
                        else:
                            classatt = AttributeInfo("classattribute", outputType.Discrete, 0, 2)

                        for datainfos in kwargs['datasetatts'].values():
                            candidateattsdict[len(candidateattsdict)] = datainfos
                        candidateattsdict[len(candidateattsdict)] = classatt
                        lock.acquire()
                        kwargs['candidateattslist'].append(candidateattsdict)
                        lock.release()
                    except Exception as ex:
                        logger.Error(f"generateTrainsetAtts", ex)

                if numofthread > 1:
                    if theproperty.mutilprocess == True:
                        mutilprocess = MutilProcess(theproperty.thread, otheroperators, opername="MLAttribute",
                                                           maxops=2000, infosep=100)
                        mutilprocess.run(myfunction, datadict=datadict, evaluator=evaluator, classifier=classifier,
                                       originalAUC=originalAUC, datasetatts=datasetatts,
                                       candidateattslist=candidateattslist)
                    else:
                        threadpool = MyThreadPool(theproperty.thread, otheroperators, opername="MLAttribute", maxops=2000, infosep=100)
                        threadpool.run(myfunction, datadict=datadict, evaluator=evaluator, classifier=classifier,
                                       originalAUC=originalAUC, datasetatts=datasetatts, candidateattslist=candidateattslist)
                else:
                    for ops in otheroperators:

                        if index % 100 == 0:
                            logger.Info("have finish " + str(index) + " / " + str(len(otheroperators)) + " operators, and time is " + str(datetime.datetime.now()))
                        if index > 2000:
                            break
                        try:
                            datacopy = copy.deepcopy(datadict)
                            candidateatt = oms.generateColumn(datacopy["data"], ops, False)
                            if candidateatt is None:
                                continue
                            obas = OperatorBasedAttributes()
                            oms.addColumn(datacopy, candidateatt)
                            candidateattsdict = obas.getOperatorsBasedAttributes(datacopy, ops, candidateatt)

                            evaluationresult = evaluator.runClassifier(classifier, datacopy)
                            auc = evaluator.calculateAUC(evaluationresult)
                            deltaAUC = auc - originalAUC
                            if deltaAUC > 0.01:
                                classatt = AttributeInfo("classattribute", outputType.Discrete, 1, 2)
                                logger.Info("find positive match")
                            else:
                                classatt = AttributeInfo("classattribute", outputType.Discrete, 0, 2)

                            for datainfos in datasetatts.values():
                                candidateattsdict[len(candidateattsdict)] = datainfos
                            candidateattsdict[len(candidateattsdict)] = classatt
                            candidateattslist.append(candidateattsdict)
                            index += 1
                        except Exception as ex:
                            logger.Error(f"generateTrainsetAtts", ex)
                            continue

        except Exception as ex:
            logger.Error(f'Failed in func "generateTrainsetAtts" with exception: {ex}')

        finally:
            if os.path.isfile(trainsetattspath) == False:
                serialize(trainsetattspath, candidateattslist)
            else:
                return deserialize(trainsetattspath)
            return candidateattslist

    def generateValuesTabular(self, dataattsvalues):
        '''

        :param dataattsvalues: [{}]
        :return: pandas.dataframe
        '''
        try:
            df = pd.DataFrame()
            attributes = self.generateAtts(dataattsvalues[0], len(dataattsvalues))
            thename = []
            for atts in attributes:
                df.insert(len(df.columns), atts.name, atts)
                thename.append(atts.name)
            num = 0
            for dav in dataattsvalues:#{1:att,2:att}
                for ats in dav.items():#(1:att)
                    index = ats[0]
                    att = ats[1]
                    if df[thename[index]].dtype in ["float", "float32", "float64"]:
                        val = (float)(att.getValue())
                    else:
                        val = (int)(att.getValue())
                    df.loc[num, thename[index]] = val
                num += 1
        except Exception as ex:
            logger.Error(f"generateValuesTabular error: {ex}")
        finally:
            return df



    def generateAtts(self, dataattsvalue, sizen: int):
        '''

        :param dataattsvalue: dict
        :return:
        '''
        attributelist = []
        for attif in dataattsvalue.values():
            # datase = [0 for _ in range(sizen)]
            if attif.getType() == outputType.Discrete:
                type = "int"
                datase = np.zeros(sizen, "int")
            elif attif.getType() == outputType.Numeric:
                type = "float"
                datase = np.zeros(sizen, "float")
            else:
                logger.Error("MLatt is not support except int and float type")

            att = pd.core.series.Series(datase, None, type, attif.getName())
            attributelist.append(att)
        return attributelist

    def getClassifierIndex(self, classifiername: str):
        index = 0
        if classifiername == "RandomForest":
            index = 0
        elif classifiername == "DicisionTree":
            index = 1
        elif classifiername == "SVM":
            index = 2
        else:
            logger.Error("No this Model : " + classifiername)
        return index