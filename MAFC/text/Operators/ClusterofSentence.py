import dask.dataframe as dd
from text.TextOperator import TextOperator
from text.cluster import processCluster
import pandas as pd
from logger.logger import logger
from properties.properties import theproperty


class ClusterofSentence(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        if theproperty.dataframe == "daks":
            datalist = list(data.iloc[:, -1].compute().values)
        elif theproperty.dataframe == "pandas":
            datalist = list(data.iloc[:, -1].values)
        else:
            logger.Info(f"no {theproperty.dataframe} can use")
        newseries = processCluster(datalist)
        series = pd.Series(newseries, name=self.getName(), dtype="int32")
        if theproperty.dataframe == "dask":
            series = dd.from_pandas(series, npartitions=1).reset_index().iloc[:, -1]
        return series

    def getName(self):
        return "ClusterofSentence"
