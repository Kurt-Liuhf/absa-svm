import copy
import pandas as pd
from image.ImageOperator import ImageOperator
from image.ImageOperators import *
from logger.logger import logger
import dask.dataframe as dd

try:
    with open(theproperty.rootpath + theproperty.extendpath, 'r') as fp:
        codes = fp.read()
    exec(codes)
except Exception as ex:
    logger.Info(f"extendpath is not exist")

def pddMerge(data1: dd.DataFrame, data2: dd.DataFrame):
    '''
    :param data1:
    :param data2:
    :return:
    '''
    for index in data2.columns:
        data1[index] = data2[index]
    return data1

def getOperatorList():
    return list(set([c.__name__ for c in ImageOperator.__subclasses__()]))

def getOperator(oplist, iglist):
    alloplist = getOperatorList()


    if oplist is not None:
        for opt in oplist:
            if opt not in alloplist:
                logger.Info(opt + " is not define")
        oplist = alloplist
    else:
        oplist = alloplist
    if iglist is not None:
        oplist = list(set(oplist) - set(iglist))
    return [eval(opt + "()") for opt in oplist]

def process(imagedata, imageoper=None, imageigoper=None):
    '''

        :param imagedata:dask.dataframe
        :param imageoper: [str]
        :param imageigoper: [str]
        :return: dask.dataframe
        '''
    #pdata = pd.DataFrame()
    data = copy.deepcopy(imagedata)
    del data[data.iloc[:, -1].name]
    operslist: list[ImageOperator] = getOperator(imageoper, imageigoper)
    for oper in operslist:
        try:
            newseries = oper.process(imagedata)
            #print(oper.getName())

            if newseries is not None:
                #newseriesview = newseries.compute()
                if type(newseries) == dd.DataFrame:
                    data = data.merge(newseries, how="left")
                elif type(newseries) == dd.core.Series:
                    data[oper.getName()] = newseries
                elif type(newseries) == pd.DataFrame:
                    data = pddMerge(data, newseries)
                elif type(newseries) == pd.Series:
                    data[oper.getName()] = newseries
        except Exception as ex:
            logger.Error(f'Failed in func "image_process" with exception: {ex}')
            continue
    data = data.set_index("id")
    return data