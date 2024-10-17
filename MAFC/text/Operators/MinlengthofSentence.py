from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class MinlengthofSentence(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getMinlength(data, nlp):
            words = nlp(data)
            lens = 1000000000
            for wd in words:
                if len(wd) < lens:
                    lens = len(wd)
            return lens

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getMinlength, nlp=nlp, meta=('getMinlength', 'i8'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getMinlength, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "MinlengthofSentence"
