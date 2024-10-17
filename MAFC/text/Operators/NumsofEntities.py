import pandas as pd
import dask.dataframe as dd
from text.TextOperator import TextOperator
from properties.properties import theproperty
from logger.logger import logger
import spacy

class NumsofEntities(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = spacy.load('en_core_web_sm')
        if theproperty.dataframe == "daks":
            datalist = list(data.iloc[:, -1].compute().values)
        elif theproperty.dataframe == "pandas":
            datalist = list(data.iloc[:, -1].values)
        else:
            logger.Info(f"no {theproperty.dataframe} can use")

        docs = list(nlp.pipe(datalist))
        newseries = [len(doc.ents) for doc in docs]
        series = pd.Series(newseries, name=self.getName(), dtype="int32")
        if theproperty.dataframe == "dask":
            series = dd.from_pandas(series, npartitions=1).reset_index().iloc[:, -1]
        return series

    def getName(self):
        return "NumsofEntities"
