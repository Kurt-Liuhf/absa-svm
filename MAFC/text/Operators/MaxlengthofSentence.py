from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class MaxlengthofSentence(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getMaxlength(data, nlp):
            words = nlp(data)
            lens = 0
            for wd in words:
                if len(wd) > lens:
                    lens = len(wd)
            return lens

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getMaxlength, nlp=nlp, meta=('getMaxlength', 'int'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getMaxlength, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "MaxlengthofSentence"
