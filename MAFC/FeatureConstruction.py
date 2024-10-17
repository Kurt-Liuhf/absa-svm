import copy
import os
import threading
from Evaluation.RankFilter.RankFilter import RankFilter
import pandas as pd

from parallel import parallel, MyThreadPool, MutilProcess
from Serialize import serialize, deserialize
from utils import *

from MAFC_Operator.OperatorManager import OperatorManager
from properties.properties import properties, theproperty


def _FC_Noiter_(datadict, unaryoperator_list: list, otheroperator_list: list, isother=False):
    '''
    非迭代
    :param datadict:{"data":dask.dataframe.Dataframe,"Info":[ColumnInfo],"target":dask.dataframe.core.Series,"targetInfo":ColumnInfo}
    :param operator_list:[operatorname: str]
    :param otheroperator_list:[operatorname: str]
    :return:dask.dataframe.Dataframe
    '''
    try:
        logger.Info("FC_Noiter begin")
        om = OperatorManager()
        #应用unary操作
        operators = om.UnaryOperator(datadict, unaryoperator_list)
        logger.Info(f"Unary finish and opertaors size is {len(operators)}")
        #将构造数据加入数据集
        om.GenerateAddColumnToData(datadict, operators, isparallel=False)
        #应用聚集操作
        if isother:
            otheroperators = om.OtherOperator(datadict, otheroperator_list)
            logger.Info(f"other finish and opertaors size is {len(operators)}")
        #将构造数据加入数据集
            om.GenerateAddColumnToData(datadict, otheroperators, isparallel=False)
    except Exception as ex:
        logger.Error(f"_FC_Noiter_ error: {ex}", ex)
    #newdata = datadict['data'].compute()
    finally:
        logger.Info(f"FC_Noiter finish and dataset size is {len(datadict['data'].columns)}")
        return datadict['data']


def _FC_Iter_(datadict, unaryoperator_list: list, otheroperator_list: list, iternums :int):
    '''
    迭代
    :param datadict:{"data":dask.dataframe.Dataframe,"Info":[ColumnInfo],"target":dask.dataframe.core.Series,"targetInfo":ColumnInfo}
    :param operator_list:[operatorname: str]
    :param otheroperator_list:[operatorname: str]
    :param iternums:int
    :return:dask.dataframe.Dataframe
    '''
    #初始化评估模型
    try:
        fevaluation = getEvaluation(theproperty.filter, datadict)
        wevaluation = getEvaluation(theproperty.wrapper, datadict)
        datasetresultpath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe\
                            + theproperty.datasetname + "result.csv"
        if theproperty.dataframe == "dask":
            datapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe \
                           + "/*"
        elif theproperty.dataframe == "pandas":
            datapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname \
                           + "/" + theproperty.dataframe + theproperty.datasetname + ".csv"
        if os.path.isfile(datasetresultpath):
            if theproperty.dataframe == "dask":
                dataset = dd.read_csv(datapath)
            elif theproperty.dataframe == "pandas":
                dataset = pd.read_csv(datapath)
            return dataset
        #进行初始评估
        #计算
        currentclassifications = wevaluation.ProduceClassifications(datadict, theproperty.classifier)
        logger.Info("Initial score is : " + str(currentclassifications.getAuc()))
        wevaluation.evaluationAsave(currentclassifications)
        wevaluation.saveModel(datadict, theproperty.classifier, "begin")
        #复制数据集
        datasetcopy = copy.deepcopy(datadict)
        om = OperatorManager()
        # 应用unary操作
        operators = om.UnaryOperator(datasetcopy, unaryoperator_list)
        # 将构造数据加入数据集
        om.GenerateAddColumnToData(datasetcopy, operators)
        if theproperty.dataframe == "dask":
            dataview = datasetcopy["data"].compute()
        #初始化排序类
        rankerFilter = RankFilter()
        #不应该重复添加
        columnaddpreiter = None
        totalnumofwrapperevaluation = 0
        finalchosenops = operators
        iters = 1
        while iters <= iternums:
            #重新计算基本特征
            try:
                # 构造特征
                # 应用聚集操作
                logger.Info(f"it is iteration of : {iters} / {iternums}")
                otheroperatorspath = theproperty.rootpath + theproperty.finalchosenopspath + theproperty.dataframe + datadict['data'].name + "otheroperators" + str(iters)
                if os.path.isfile(otheroperatorspath):
                    otheroperators = deserialize(otheroperatorspath)
                else:
                    #logger.Info(getNowTimeStr())
                    fevaluation.recalculateDatasetBasedFeatures(datasetcopy)
                    #logger.Info(getNowTimeStr())
                    otheroperators = om.OtherOperator(datasetcopy, otheroperator_list)
                    # 重新使用F计算分数
                    otheroperators = list(set(otheroperators) - set(finalchosenops))
                    logger.Info("getfsocre start " + getNowTimeStr())
                    om.reCalcFEvaluationScores(datasetcopy, otheroperators, fevaluation)
                    logger.Info("getfsocre end " + getNowTimeStr())
                    #排序
                    otheroperators = rankerFilter.rankAndFilter(otheroperators, columnaddpreiter)
                    serialize(otheroperatorspath, otheroperators)
                finaldatapath = theproperty.rootpath + theproperty.finalchosenopspath + theproperty.dataframe + \
                                datadict['data'].name + "_data_" + str(iters)
                finalchosenopspath = theproperty.rootpath + theproperty.finalchosenopspath + theproperty.dataframe + \
                                     datadict['data'].name + "_chosenoperators_" + str(iters)

                evaluationatts = [0]
                chosenoperators = None
                toprankingoperators = []
                tempcurrentclassifications = currentclassifications
                #使用W计算分数,并行计算
                numofthread = theproperty.thread
                logger.Info("getwsocre start " + getNowTimeStr())
                lock = threading.Lock()
                def myevaluationfunc(oop, **kwargs):
                    try:

                        if oop.getFScore() is not None and oop.getFScore() >= theproperty.fsocre:
                            kwargs['evaluationatts'][0] += 1
                            datacopy = copy.deepcopy(kwargs['datasetcopy'])
                            newcolumn = om.generateColumn(datacopy["data"], oop, False)
                            if newcolumn is None:
                                return 0
                            wscore = kwargs['wevaluation'].produceScore(datacopy, kwargs['tempcurrentclassifications'], oop, newcolumn)
                            oop.setWScore(wscore)
                            if theproperty.wsocre <= wscore:
                                lock.acquire()
                                kwargs['toprankingoperators'].append(oop)
                                lock.release()
                            return wscore
                    except Exception as ex:
                        logger.Error(f"FC WEvaluation error: {ex}", ex)
                        return 0

                if numofthread == 1:
                    count = 0
                    for oop in otheroperators:
                        try:
                            if count >= theproperty.maxevaluationattsperiter:
                                break
                            if(oop.getFScore() is not None and oop.getFScore() >= theproperty.fsocre and evaluationatts[0] <= theproperty.maxevaluationattsperiter):
                                datacopy = copy.deepcopy(datasetcopy)
                                newcolumn = om.generateColumn(datacopy["data"], oop, False)
                                if newcolumn is None:
                                    return 0
                                wscore = wevaluation.produceScore(datacopy, tempcurrentclassifications, oop, newcolumn)
                                oop.setWScore(wscore)
                                evaluationatts[0] += 1
                                if theproperty.wsocre <= wscore:
                                    #logger.Info("chosen a operator :" + oop.getName())
                                    toprankingoperators.append(oop)

                                if(evaluationatts[0] % 100 == 0):
                                    logger.Info("evaluated " + str(evaluationatts[0]) + " / " + str(min(len(otheroperators), theproperty.maxevaluationattsperiter)) + " attributes, and time is " + str(datetime.datetime.now()))
                            else:
                                break
                        except Exception as ex:
                            logger.Error(f"FC WEvaluation error: {ex}", ex)
                            continue

                else:
                    if theproperty.mutilprocess == True:
                        mutilprocess = MutilProcess(theproperty.thread, otheroperators, opername="WEvaluation", infosep=100, maxops=theproperty.maxevaluationattsperiter)
                        res = mutilprocess.run(myfuncforwscore, datasetcopy=datasetcopy, wevaluation=wevaluation,
                                       tempcurrentclassifications=tempcurrentclassifications)
                        evaluationatts[0] += len(res)
                        for socre, ops in zip(res, otheroperators):
                            wsocre = socre.get()
                            ops.setWScore(wsocre)
                            if theproperty.wsocre <= wsocre:
                                #logger.Info("chosen a operator :" + oop.getName())
                                toprankingoperators.append(ops)
                    else:
                        threadpool = MyThreadPool(theproperty.thread, otheroperators, opername="WEvaluation", infosep=100, maxops=theproperty.maxevaluationattsperiter)
                        threadpool.run(myevaluationfunc, datasetcopy=datasetcopy, wevaluation=wevaluation, tempcurrentclassifications=tempcurrentclassifications,
                                       toprankingoperators=toprankingoperators, evaluationatts=evaluationatts)

                logger.Info("getwsocre end " + getNowTimeStr())
                # 筛选特征
                totalnumofwrapperevaluation += evaluationatts[0]
                if chosenoperators is None:
                    if len(toprankingoperators) > 0:
                        toprankingoperators = rankerFilter.rankAndWrapper(toprankingoperators, columnaddpreiter)
                        if len(toprankingoperators) > 10:
                            chosenoperators = copy.deepcopy(toprankingoperators[0:10])
                        else:
                            chosenoperators = copy.deepcopy(toprankingoperators)
                    else:
                        logger.Info("No attributes are chosen,iteration over!")
                        break
                finalchosenops += chosenoperators

                serialize(finalchosenopspath, chosenoperators)
                om.GenerateAddColumnToData(datasetcopy, chosenoperators)
                om.deleteColumn(datasetcopy)
                currentclassifications = wevaluation.ProduceClassifications(datasetcopy, theproperty.classifier)
                wevaluation.evaluationAsave(currentclassifications, iters, chosenoperators, totalnumofwrapperevaluation, False)

                serialize(finaldatapath, datasetcopy)

            except Exception as ex:
                logger.Error(f"calculateWsocre while error!{ex}")
            finally:
                iters += 1
        wevaluation.saveModel(datasetcopy, theproperty.classifier, "end")
        serialize(theproperty.rootpath + theproperty.finalchosenopspath + theproperty.dataframe + theproperty.datasetname
                  + "finalchosenops", finalchosenops)
        return datasetcopy['data']
    except Exception as ex:
        logger.Error(f"_FC_Iter_ error!{ex}")

def myfuncforwscore(oop, kwargs):
    try:
        om = OperatorManager()
        if oop.getFScore() is not None and oop.getFScore() >= theproperty.fsocre:
            datacopy = copy.deepcopy(kwargs['datasetcopy'])
            newcolumn = om.generateColumn(datacopy["data"], oop, False)
            if newcolumn is None:
                return 0
            wscore = kwargs['wevaluation'].produceScore(datacopy, kwargs['tempcurrentclassifications'], oop, newcolumn)
            return wscore
    except Exception as ex:
        logger.Error(f"FC WEvaluation error: {ex}", ex)
        return 0

def _FC_(datadict, isiteration: bool = False, iternums: int = 1, operatorbyself: dict = None, operatorignore: dict = None, isother = False):
    '''

    :param datadict: {"data":dask.dataframe.Dataframe,"Info":[ColumnInfo],"target":dask.dataframe.core.Series,"targetInfo":ColumnInfo}
    :param isiteration: bool
    :param iternums: int
    :param operatorbyself: {"text":[],"tabular":{'unary':[],'other':[]},"image":[]}自定义操作
    :param operatorignore:{"text":[],"tabular":{'unary':[],'other':[]},"image":[]}忽略的操作
    :return:dask.dataframe.Dataframe
    '''
    try:
        unaryoperatorlist = properties().unaryoperator
        otheroperatorlist = properties().otheroperator
        unaryoperator_list = unaryoperatorlist.copy()
        otheroperator_list = otheroperatorlist.copy()
        # 选择操作列表
        if operatorbyself is not None and operatorbyself.get("tabular") is not None:
            if operatorbyself["tabular"].get("unary") is not None:
                unaryoperator_list = unaryoperator_list + operatorbyself["tabular"]['unary']
            if operatorbyself["tabular"].get("other") is not None:
                otheroperator_list = otheroperator_list + operatorbyself["tabular"]['other']
        if operatorignore is not None and operatorignore.get("tabular") is not None:
            if operatorignore["tabular"].get("unary") is not None:
                unaryoperator_list = list(set(unaryoperator_list) - set(operatorignore["tabular"]['unary']))
            if operatorignore["tabular"].get("other") is not None:
                otheroperator_list = list(set(otheroperator_list) - set(operatorignore["tabular"]['other']))
        df = None

        if isiteration is False:
            df = _FC_Noiter_(datadict, unaryoperator_list, otheroperator_list, isother=isother)
        else:
            df = _FC_Iter_(datadict, unaryoperator_list, otheroperator_list, iternums)
    except Exception as ex:
        logger.Error(f'Failed in func "_FC_" with exception: {ex}')
    finally:
        # saveDateFrame(df, datadict["data"].name)
        return df

def FC(dataset, isiteration: bool = False, iternums: int = 1,operatorbyself: dict = None, operatorignore: dict = None, isother = False):
    '''
    :param operatorignore:{"text":[],"tabular":{'unary':[],'other':[]},"image":[]}自定义操作
    :param operatorbyself:{"text":[],"tabular":{'unary':[],'other':[]},"image":[]}忽略的操作
    :param iternums: 迭代次数
    :param dataset:数据集
    :param isiteration: 是否进行迭代
    :return: dask.dataframe.Dataframe
    '''
    datadictpath = theproperty.rootpath + theproperty.temppath + theproperty.dataframe + theproperty.datasetname + "datadict.temp"
    # if os.path.isfile(datadictpath):
    #     datadict = deserialize(datadictpath)
    # else:
    datadict = getDatadict(dataset, operatorbyself, operatorignore)
        # serialize(datadictpath, datadict)
    datadict["data"].name = theproperty.datasetname
    df = _FC_(datadict, isiteration, iternums, operatorbyself, operatorignore, isother=isother)
    return df

def generateTestData(dataset, datasettrainname=theproperty.datasetname):
    datadictpath = theproperty.rootpath + theproperty.temppath + theproperty.dataframe + datasettrainname + "testdatadict.temp"
    logger.Info("strat generateTestData!")
    resultpath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + datasettrainname + "testresult.csv"

    if theproperty.dataframe == "dask":
        datapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + "/test" \
                       + "/*"
    elif theproperty.dataframe == "pandas":
        datapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "test" \
                       + "/" + theproperty.dataframe + theproperty.datasetname + "test.csv"
    if os.path.isfile(resultpath):
        if theproperty.dataframe == "dask":
            dataset = dd.read_csv(datapath)
        elif theproperty.dataframe == "pandas":
            dataset = pd.read_csv(datapath)
        return dataset
    if os.path.isfile(datadictpath):
        datadict = deserialize(datadictpath)
    else:
        datadict = getDatadict(dataset)
        serialize(datadictpath, datadict)
    saveDateFrame(datadict["data"], theproperty.datasetname + "testoriginal")
    operators = deserialize(theproperty.rootpath + theproperty.finalchosenopspath + theproperty.dataframe + datasettrainname + "finalchosenops")
    om = OperatorManager()
    #初始评估
    wevaluation = AucWrapperEvaluation()
    #datacopy = copy.deepcopy(datadict)
    classificationresult = wevaluation.getTestClassifications(datadict, theproperty.classifier, "begin")
    wevaluation.evaluationAsave(classificationresult, 0, istest=True)
    om.GenerateAddColumnToData(datadict, operators, isparallel=False)
    processTestData(datadict)

    classificationresult = wevaluation.getTestClassifications(datadict, theproperty.classifier, "end")
    wevaluation.evaluationAsave(classificationresult, 1, operators, len(operators), newfile=False, istest=True)
    saveDateFrame(datadict["data"], theproperty.datasetname + "test")
    return datadict["data"]



