from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class CountWords(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getCountWords(data, nlp):
            words = nlp(data)
            return len(words)

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getCountWords, nlp=nlp, meta=('getCountWords', 'int'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getCountWords, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "CountWords"
