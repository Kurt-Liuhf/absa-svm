import os

import pandas as pd

from Evaluation.WEvaluation.FoneEvaluation import FoneEvaluation
from Evaluation.WEvaluation.LogLossEvaluation import LogLossEvaluation
from Evaluation.FEvaluation.InformationGain import InformationGainFilterEvaluator
from Evaluation.FEvaluation.MLFEvalution import MLFEvaluation
from Evaluation.WEvaluation.AucWrapperEvaluation import AucWrapperEvaluation
from MAFC_Operator.ColumnInfo import ColumnInfo
from MAFC_Operator.operator_base import outputType
from Serialize import serialize
from logger.logger import logger
from text.text_process import process as txprocess
from image.image_process import process as improcess
from properties.properties import theproperty
import dask.dataframe as dd
import datetime
from Dataset import Dataset
from Evaluation.FEvaluation.MLAttributeManager import MLAttributeManager
def Image_FC(image_data, imageoper, imageigoper):
    '''
    处理Image数据
    :param image_data:
    :return: dask.Dataframe
    '''
    if image_data is None:
        return None
    return improcess(image_data, imageoper, imageigoper)

def Text_FC(text_data, textoper, textigoper):
    '''
    处理Text数据
    :param text_data:
    :return: dask.Dataframe
    '''
    if text_data is None:
        return None
    return txprocess(text_data, textoper, textigoper)

def getInfo(data):
    '''
    为数据生成ColumnInfo
    :param data: dask.dataframe.Dataframe
    :return: [ColumnInfo]
    '''
    info = []
    lenofdata = len(data)
    targetname = data.iloc[:, -1].name
    for i in data:
        name = i
        if data[name].dtype == "datetime64[ns]":
            columninfo = ColumnInfo(None, None, None, name, False, outputType.Date)
            info.append(columninfo)
            continue
        elif data[name].dtype == "float64" or data[name].dtype == "float32":
            columninfo = ColumnInfo(None, None, None, name, False, outputType.Numeric)
            info.append(columninfo)
            continue
        if theproperty.dataframe == "dask":
            datavalues = data[name].value_counts().compute()
        else:
            datavalues = data[name].value_counts()
        lensofvalues = len(datavalues)
        if (lenofdata == lensofvalues or lensofvalues == 1):
            del data[name]
            continue

        if data[name].dtype == "int64" or data[name].dtype == "int32":
            if lensofvalues >= 30 and name != targetname:
                data[name].astype("float32")
                columninfo = ColumnInfo(None, None, None, name, False, outputType.Numeric)
            else:
                seriesdict = {}
                dictnum = 0
                for thekey in datavalues.keys():
                    seriesdict[thekey] = dictnum
                    dictnum += 1
                # data[name].compute()
                data[name] = data[name].replace(to_replace=seriesdict)
                columninfo = ColumnInfo(None, None, None, name, False, outputType.Discrete, lensofvalues)
        elif data[name].dtype == "object":
            seriesdict = {}
            dictnum = 0
            for thekey in datavalues.keys():
                seriesdict[thekey] = dictnum
                dictnum += 1
            # data[name].compute()
            data[name] = data[name].replace(to_replace=seriesdict)
            data[name].astype("int32")
            columninfo = ColumnInfo(None, None, None, name, False, outputType.Discrete, lensofvalues)
        elif data[name].dtype == "bool":
            data[name] = data[name].replace({"False": 0, "True": 1, "false": 0, "true": 1})
            # data[name].compute()
            data[name].astype("int32")
            columninfo = ColumnInfo(None, None, None, name, False, outputType.Discrete, 2)
        info.append(columninfo)
    logger.Info("getInfo complete")
    return info

def getEvaluation(name, datadict):
    '''
    获取评估器
    :param name: str ,评估器的名字
    :param datadict: {"data":dask.dataframe.Dataframe,"Info":[ColumnInfo],"target":dask.dataframe.core.Series,"targetInfo":ColumnInfo}
    :return: Evaluation
    '''
    evaluation = None
    if name == "AucWrapperEvaluation":
        evaluation = AucWrapperEvaluation()
    elif name == "InformationGain":
        evaluation = InformationGainFilterEvaluator()
    elif name == "LogLossEvaluation":
        evaluation = LogLossEvaluation()
    elif name == "MLFEvaluation":
        evaluation = MLFEvaluation(datadict)
    elif name == "FoneEvaluation":
        evaluation = FoneEvaluation()
    else:
        logger.Error("No this Evaluation" + name)
    return evaluation

def Merge_Data(image_data, text_data, tab_data):
    '''
    :param image_data:dask.dataframe.Dataframe
    :param text_data:dask.dataframe.Dataframe
    :param tab_data:dask.dataframe.Dataframe
    :return:dask.dataframe.Dataframe
    '''
    data = None
    if theproperty.dataframe == "pandas":
        if image_data is not None:
            data = image_data
            if text_data is not None:
                data = pddMerge(data, text_data)
                if tab_data is not None:
                    data = pddMerge(data, tab_data)
            elif tab_data is not None:
                data = pddMerge(data, tab_data)
        else:
            if text_data is not None:
                data = text_data
                if tab_data is not None:
                    data = pddMerge(data, tab_data)
            elif tab_data is not None:
                data = tab_data
        if data is None:
            logger.Info(f"Error, No data!")
        data.name = theproperty.datasetname
        return data
    if image_data is not None:
        data = image_data
        if text_data is not None:
            data = data.merge(text_data.compute(), how="outer")
            if tab_data is not None:
                data = data.merge(tab_data.compute(), how="outer")
        elif tab_data is not None:
            data = data.merge(tab_data.compute(), how="outer")
    else:
        if text_data is not None:
            data = text_data
            if tab_data is not None:
                data = data.merge(tab_data.compute(), how="outer")
        elif tab_data is not None:
            data = tab_data
    if data is None:
        logger.Error(f"No data")

    thetime = getNowTimeStr()
    if data is not None:
        tempimagedir = theproperty.temppath + theproperty.datasetname + "_" + thetime
        os.makedirs(tempimagedir)
        data.to_csv(tempimagedir, index=False)
        data = dd.read_csv(tempimagedir + "/*.part")
    data.name = theproperty.datasetname
    return data

def getDatadict(dataset, operatorbyself=None, operatorignore=None, haslabel = True):
    '''

    :param dataset: Dataset
    :param operatorbyself: {"text":[],"tabular":[],"image":[]}
    :param operatorignore: {"text":[],"tabular":[],"image":[]}
    :return: {"data":dask.dataframe.Dataframe,"Info":[ColumnInfo],"target":dask.dataframe.core.Series,"targetInfo":ColumnInfo}
    '''
    textoper = None
    imageoper = None
    textigoper = None
    imageigoper = None
    if operatorbyself is not None:
        if operatorbyself.get("text") is not None:
            textoper = operatorbyself["text"]
        if operatorbyself.get("image") is not None:
            imageoper = operatorbyself["image"]
    if operatorignore is not None:
        if operatorignore.get("text") is not None:
            textigoper = operatorignore["text"]
        if operatorignore.get("image") is not None:
            imageigoper = operatorignore["image"]
    logger.Info(f"image_fc begin ,time is {datetime.datetime.now()}")
    image_fc = Image_FC(dataset.data_image, imageoper, imageigoper)
    logger.Info(f"image_fc end ,time is {datetime.datetime.now()}\ntext_fc begin ,time is {datetime.datetime.now()}")
    text_fc = Text_FC(dataset.data_text, textoper, textigoper)
    logger.Info(f"text_fc end ,time is {datetime.datetime.now()}")
    ##合并image、text和tabular
    if theproperty.dataframe == "dask":
        image_fc, text_fc = saveData(image_fc, text_fc)
    data = Merge_Data(image_fc, text_fc, dataset.data_tabular)

    #textview = text_fc.compute()
    #imageview = image_fc.compute()
    #dataview = data.compute()
    saveDateFrame(data, theproperty.datasetname + "original")
    datainfo = getInfo(data)

    datadict = {"data": data, "Info": datainfo}
    index = theproperty.targetindex
    datadict["target"] = datadict["data"].iloc[:, index]
    del datadict["data"][datadict["target"].name]
    if haslabel is True:
        datadict["targetInfo"] = datainfo.pop(-1)
        theproperty.targetclasses = datadict["targetInfo"].numsofunique
    if theproperty.targetclasses is not None and theproperty.targetclasses > 2:
        theproperty.targetmutil = True
    return datadict

def saveData(image_fc: dd.DataFrame, text_fc: dd.DataFrame):
    '''
    保存image和text生成的中间特征数据
    :param image_fc: dd.DataFrame
    :param text_fc: dd.DataFrame
    :return: dd.DataFrame , dd.DataFrame
    '''
    thetime = getNowTimeStr()
    if image_fc is not None:
        tempimagedir = theproperty.temppath + "image_" + thetime
        os.makedirs(tempimagedir)
        image_fc.to_csv(tempimagedir, index=False)
        image_fc = dd.read_csv(tempimagedir + "/*.part")
        #imageview = image_fc.compute()
    if text_fc is not None:
        temptextdir = theproperty.temppath + "text_" + thetime
        os.makedirs(temptextdir)
        text_fc.to_csv(temptextdir, index=False)
        text_fc = dd.read_csv(temptextdir + "/*.part")
        #textview = text_fc.compute()
    return image_fc, text_fc

def generateModelData(dataset: Dataset):
    '''
    该函数用于生成过滤器模型所需的数据，请先利用其它数据集创建Dataset(),设置好theproperty.datasetlocation
    :return:
    '''
    datadict = getDatadict(dataset)
    mlam = MLAttributeManager()
    mlam.getDatasetInstances(datadict)

def getNowTimeStr():
    thetime = str(datetime.datetime.now()).replace(" ", "").replace("-", "").replace(":", "").replace(".", "")
    return thetime

def saveDateFrame(dataframe: dd.DataFrame, name: str = theproperty.datasetname):
    try:
        if dataframe is not None:
            #serialize(theproperty.resultfilepath + theproperty.dataframe + name + ".datatemp", dataframe)
            newdir = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + name
            if os.path.isdir(newdir) == False:
                os.mkdir(newdir)

            if theproperty.dataframe == "dask":
                dataframe.to_csv(newdir, index=False)
            elif theproperty.dataframe == "pandas":
                dataframe.to_csv(newdir + os.path.sep + theproperty.dataframe + name + ".csv", index=False)
    except Exception as ex:
        pass

def pddMerge(data1: dd.DataFrame, data2: dd.DataFrame):
    '''
    :param data1:
    :param data2:
    :return:
    '''
    for index in data2.columns:
        data1[index] = data2[index]
    return data1

def processTestData(datadict, ):
    testdata = datadict["data"]
    if theproperty.dataframe == "pandas":
        traindatapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname\
                    + "/" + theproperty.dataframe + theproperty.datasetname + ".csv"
        traindata = pd.read_csv(traindatapath)
    else:
        traindatapath = theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname \
                        + "/*"
        traindata = dd.read_csv(traindatapath)
    namelist = list(set(testdata) - set(traindata))
    for name in namelist:
        del testdata[name]
    i = 0
    for name in list(traindata.columns):
        tempcolumns = testdata[name]
        del testdata[name]
        testdata.insert(i, tempcolumns.name, tempcolumns)
        i = i + 1
    datadict["data"] = testdata.copy()

def execmyfunc(codes):
    exec(codes)
