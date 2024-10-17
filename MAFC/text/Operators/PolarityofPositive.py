import pandas as pd
import dask.dataframe as dd
from text.TextOperator import TextOperator
from text.lexicon import lexicons_features_vectors
from text.stanford_nlp import StanfordNLP
from logger.logger import logger
from properties.properties import theproperty


class PolarityofPositive(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        def pol(resvec):
            res = resvec[4]
            if res > 2:
                return 1
            else:
                return 0

        if theproperty.dataframe == "daks":
            datalist = list(data.iloc[:, -1].compute().values)
        elif theproperty.dataframe == "pandas":
            datalist = list(data.iloc[:, -1].values)
        else:
            logger.Info(f"no {theproperty.dataframe} can use")

        nlp_helper = StanfordNLP()
        words = []
        pos_tags = []
        for text in datalist:
            word, pos_tag = nlp_helper.pos_tag(text)
            words.append(word)
            pos_tags.append(pos_tag)
        wordvec = lexicons_features_vectors(words,pos_tags)
        newseries = [pol(i) for i in wordvec]
        series = pd.Series(newseries, name=self.getName(), dtype="int32")
        if theproperty.dataframe == "dask":
            series = dd.from_pandas(series, npartitions=1).reset_index().iloc[:, -1]
        return series

    def getName(self):
        return "PolarityofPositive"
