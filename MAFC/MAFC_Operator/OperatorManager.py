import copy
import datetime
import threading
from random import random

import numpy as np
from MAFC_Operator.Operators import Operators
from logger.logger import logger
from parallel import parallel, MyThreadPool, MutilProcess
from properties.properties import theproperty
import pandas as pd
from MAFC_Operator.ColumnInfo import ColumnInfo
from MAFC_Operator.Combination import Combination
from MAFC_Operator.operator_base import operatorType
from MAFC_Operator.Binary import *
from MAFC_Operator.Unary import *
from MAFC_Operator.Groupby import *
try:
    with open(theproperty.rootpath + theproperty.extendpath, 'r') as fp:
        codes = fp.read()
    exec(codes)
except Exception as ex:
    logger.Info(f"extendpath is not exist")
    


class OperatorManager:
    def __init__(self):
        pass
    def addColumn(self, datadict, candidateatt):
        '''
        :param datadict:
        :param candidateatt:(name,data)
        :return:
        '''
        if candidateatt is None or candidateatt[1] is None:
            return
        try:
            # if candidateatt[0] in datadict['data'].columns:
            #     logger.Info(f"{candidateatt[0]} have existed in data")
            datadict["data"][candidateatt[0]] = candidateatt[1]
            if candidateatt[0] in datadict["data"].columns:
                datadict["Info"].append(candidateatt[2])
            else:
                pass
                #logger.Info(f"{candidateatt[0]} addcolumn isn't successful")
        except Exception as ex:
            logger.Error(f"addColumn Error: {ex}", ex)

    def deleteColumn(self, datadict, dellist=None):
        try:
            if dellist is not None:
               del datadict["data"][dellist]
            listname = datadict['data'].columns
            newinfo = list(set(filter(lambda info: info.getName() in listname, datadict["Info"])))
            datadict['Info'] = newinfo
        except Exception as ex:
            logger.Error(f"deleteColumn error: {ex}", ex)

    def UnaryOperator(self, data, unaryoperatorlist):
        '''
        :param data: {"data":data,"Info":[ColumnInfo]}
        :param unaryoperatorlist: ["operator"]
        :return: [Operators]
        '''
        unaryoperators = [self.getUnaryOperator(unaryoperator) for unaryoperator in unaryoperatorlist]
        operators = self.getOperators(data, unaryoperators, 1)
        logger.Info("UnaryOperator complete")
        return operators

    def generateColumn(self, data, ops: Operators, needcompute: bool = False):
        '''
        :param data: data
        :param os:Operators
        :return: [values]
        '''
        try:
            operator = ops.getOperator()
            scdict = []
            if ops.sourceColumns is not None:
                scdict = [{"name": sc.getName(), "type": sc.getType()} for sc in ops.sourceColumns]
            tcdict = []
            if ops.targetColumns is not None:
                tcdict = [{"name": tc.getName(), "type": tc.getType()} for tc in ops.targetColumns]
            operator.processTrainingSet(data, scdict, tcdict)
            newcolumn = operator.generateColumn(data, scdict, tcdict)
            if newcolumn["data"] is None:
                logger.Info(f"{operator.getName()} generateColumn is None!")
            else:
                newcolumndata = newcolumn["data"].fillna(0)

            if theproperty.dataframe != "dask":
                needcompute = False
            if needcompute == True:
                newcolumndata = newcolumndata.compute()
            if ops.getType() == outputType.Discrete:
                lensofvalues = ops.getNumofBins()
                columninfo = ColumnInfo(ops.sourceColumns, ops.targetColumns, operator, ops.getName(), False, ops.getType(), lensofvalues)
            else:
                columninfo = ColumnInfo(ops.sourceColumns, ops.targetColumns, operator, ops.getName(), False, ops.getType())
            return [ops.getName(), newcolumndata, columninfo]
        except Exception as ex:
            logger.Error(f"generateColumn error: {ops.getOperator().getName()} , {ex}", ex)
            return None


    def GenerateAddColumnToData(self, datadict, operators, isparallel=True):
        '''
        :param datadict:{"data":data,"Info":[ColumnInfo]}
        :param operators: [Operators]
        :return: Null
        '''
        try:
            lock = threading.Lock()
            def myfunc(ops, **kwargs):
                #logger.Info(f'{ops.getName()} is start work')
                newcolumn = self.generateColumn(kwargs['datadict']["data"], ops)
                if newcolumn is None:
                    return
                kwargs['lock'].acquire()
                self.addColumn(kwargs['datadict'], newcolumn)
                #logger.Info(f"{newcolumn[0]} add successful")
                kwargs['lock'].release()

            threadnums = theproperty.thread
            if threadnums == 1 or isparallel is False :
                osnums = len(operators)
                num = 1
                for ops in operators:
                    if num % 1000 == 0:
                        logger.Info("this is " + str(num) + " / " + str(osnums) + " and time is " + str(datetime.datetime.now()))
                    num += 1
                    newcolumn = self.generateColumn(datadict["data"], ops)
                    if newcolumn is None:
                        continue
                    self.addColumn(datadict, newcolumn)
            else:
                threadpool = MyThreadPool(threadnums, operators, opername="AddData")
                threadpool.run(myfunc, datadict=datadict, lock=lock)

            logger.Info("GenerateAddColumnToData complete")
        except Exception as ex:
            logger.Error(f"GenerateAddColumnToData error: {ex}", ex)

    def OtherOperator(self, data, operatorlist):
        '''
        :param data: {"data":data,"Info":[ColumnInfo]}
        :param operatorlist: ["operator"]
        :return: [Operators]
        '''
        otheroperators = [self.getOtherOperator(operator) for operator in operatorlist]
        operators = self.getOperators(data, otheroperators, theproperty.maxcombination)
        logger.Info("OtherOperator complete")
        return operators

    def getCombination(self, attributes: list, numsofcombination):
        lengthofattribute = len(attributes)
        comb = Combination(lengthofattribute, numsofcombination)
        attributecombination = []
        while comb.hasMore():
            indices = comb.getNext()
            thecolumns = []
            for index in indices:
                thecolumns.append(attributes[index])
            attributecombination.append(thecolumns.copy())
        return attributecombination

    def getUnaryOperator(self,name):
        ob = eval(name + "()")
        return ob

    def getOtherOperator(self, name):
        if "Time" in name:
            names = name.split("_")
            ob = eval(names[0] + "(" + names[1] + ")")
        else:
            ob = eval(name + "()")
        return ob




    def getOperator(self, oper):
        if oper.getType() == operatorType.Unary:
            return self.getUnaryOperator(oper.getName())
        return self.getOtherOperator(oper.getName())

    def overlapexists(self, sources, target):
        '''
        :param sources:[ColumnInfo]
        :param target: ColumnInfo
        :return:bool
        '''
        #首先检查是否包含相同的
        if target in sources:
            return True
        #然后检查是否包含共享的
        sourcesans = []
        for source in sources:
            sourcesans.append(source)
            if source.getSourceColumns() is not None:
                for ansource in source.getSourceColumns():
                    if ansource not in sourcesans:
                        sourcesans.append(ansource)
            if source.getTargetColumns() is not None:
                for antarget in source.getTargetColumns():
                    if antarget not in sourcesans:
                        sourcesans.append(antarget)

        targetans = []
        targetans.append(target)
        if target.getSourceColumns() is not None:
            for ansource in target.getSourceColumns():
                if ansource not in targetans:
                    targetans.append(ansource)
        if target.getTargetColumns() is not None:
            for antarget in target.getTargetColumns():
                if antarget not in targetans:
                    targetans.append(antarget)
        overlap = len(set(sourcesans) & set(targetans)) > 0
        '''
        if overlap and len(targetans) > 1:
            raise ("target and sources overlap error!")
        '''
        return overlap

    def getUnaryOperatorList(self):
        operatorlist = []
        for op in theproperty.unaryoperator:
            operatorlist.append(self.getUnaryOperator(op))
        return operatorlist

    def getOperators(self, data, operatorlist, maxcombinations, includeattributes = None):
        '''
        :param includeattributes:[ColumnInfo]
        :param operatorlist:['operator']
        :param data:{"data":data,"Info":[ColumnInfo]}
        :return:the list of Operators
        '''

        if includeattributes == None:
            includeattributes = []
        theoperators = []
        i = maxcombinations
        while i > 0:
            if len(theoperators) > theproperty.maxoperators:
                break
            attributecombination = self.getCombination(data["Info"], i)
            for ac in attributecombination:
                if len(theoperators) > theproperty.maxoperators:
                    break
                if len(includeattributes) > 0:
                    thecolumn = copy.deepcopy(ac)
                    thecolumn = list(set(thecolumn) & set(includeattributes))
                    if len(thecolumn) == 0:
                        continue
                for op in operatorlist:
                    if len(theoperators) > theproperty.maxoperators:
                        break
                    if(op.isMatch(data["data"],[{"name":itac.getName(),"type":itac.getType()}for itac in ac],[])):
                        newops = Operators(ac, None, self.getOperator(op), None)
                        theoperators.append(newops)

                    for tc in list(data["Info"]):
                        if len(theoperators) > theproperty.maxoperators:
                            break
                        if self.overlapexists(ac, tc):
                            continue
                        thecolumn = []
                        thecolumn.append(tc)
                        if(op.isMatch(data["data"],[{"name":itac.getName(),"type":itac.getType()}for itac in ac],[{"name":tc.getName(),"type":tc.getType()}for tc in thecolumn])):
                            ops = Operators(ac, thecolumn, self.getOperator(op), None)
                            theoperators.append(ops)
            i = i - 1
        addoperators = []
        for os in theoperators:
            if len(theoperators) + len(addoperators) > theproperty.maxoperators:
                break
            if os.getOperator().getType() != operatorType.Unary:
                for oper in self.getUnaryOperatorList():
                    if len(theoperators) + len(addoperators) > theproperty.maxoperators:
                        break
                    if(oper.getType() == os.getOperator().getOutputType()):
                        addops = Operators(os.sourceColumns, os.targetColumns, os.operators, oper)
                        addoperators.append(addops)
        theoperators = theoperators + addoperators
        return theoperators

    def reCalcFEvaluationScores(self, datadict, operators, fevaluation):
        if fevaluation.needToRecalcScore() is False:
            return
        else:
            self.calculateFsocre(datadict, fevaluation, operators)

    def calculateFsocre(self, datadict, fevaluation, operators):

        numOfThread = theproperty.thread
        count = 0

        def myfunction(ops, **kwargs):
            try:
                datacopy = copy.deepcopy(kwargs['datadict'])
                newcolumn = self.generateColumn(datacopy["data"], ops)
                if newcolumn is None or kwargs['fevaluation'] is None:
                    logger.Info("generate column or fevaluation error!")
                    return
                newfevaluation = copy.deepcopy(kwargs['fevaluation'])
                templist = [newcolumn]
                newfevaluation.initFEvaluation(templist)
                fsocre = newfevaluation.produceScore(datacopy, None, ops, newcolumn)
                ops.setFScore(fsocre)
            except Exception as ex:
                logger.Error(f"calculateFsocre error!{ex}")
                ops.setFScore(0.0)

        if numOfThread == 1:
            for ops in operators:
                try:
                    if count > theproperty.maxFEvaluationnums:
                        break
                    count += 1
                    if count % 1000 == 0:
                        logger.Info(f"analyzed {count} / {min(len(operators), theproperty.maxFEvaluationnums)} attributes, and time is {datetime.datetime.now()}")
                    newcolumn = self.generateColumn(datadict["data"], ops)
                    if newcolumn is None or fevaluation is None:
                        logger.Info("generate column or fevaluation error!")
                        continue
                    datacopy = copy.deepcopy(datadict)
                    newfevaluation = copy.deepcopy(fevaluation)
                    self.addColumn(datacopy, newcolumn)
                    #templist = [newcolumn]
                    #newfevaluation.initFEvaluation(templist)
                    fsocre = newfevaluation.produceScore(datacopy, None, ops, newcolumn)
                    ops.setFScore(fsocre)
                except Exception as ex:
                    logger.Error(f"calculateFsocre error!{ex}")
                    ops.setFScore(0.0)
        else:
            #parallel.ParallelForEachShare(myfunction, [ops for ops in operators])
            if theproperty.mutilprocess == True:
                mutilprocess = MutilProcess(theproperty.thread, operators, theproperty.maxFEvaluationnums,
                                          opername="FEvaluation")
                res = mutilprocess.run(myfuncforfsocre, datadict=datadict, fevaluation=fevaluation)
                for socre, ops in zip(res, operators):
                    ops.setFScore(socre.get())
            else:
                threadpool = MyThreadPool(theproperty.thread, operators, theproperty.maxFEvaluationnums, opername="FEvaluation")
                threadpool.run(myfunction, datadict=datadict, fevaluation=fevaluation)

def myfuncforfsocre(ops, kwargs):
    try:
        om = OperatorManager()
        datacopy = copy.deepcopy(kwargs['datadict'])
        newcolumn = om.generateColumn(datacopy["data"], ops)
        if newcolumn is None or kwargs['fevaluation'] is None:
            logger.Info("generate column or fevaluation error!")
            return
        newfevaluation = copy.deepcopy(kwargs['fevaluation'])
        templist = [newcolumn]
        newfevaluation.initFEvaluation(templist)
        fsocre = newfevaluation.produceScore(datacopy, None, ops, newcolumn)
        return fsocre
    except Exception as ex:
        logger.Error(f"calculateFsocre error!{ex}")
        return 0.0

