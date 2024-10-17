from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class MaxlengthWord(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getMaxlengthWord(data, nlp):
            words = nlp(data)
            word = None
            for wd in words:
                if word == None:
                    word = wd
                    continue
                if len(wd) > len(word):
                    word = wd
            return word

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getMaxlengthWord, nlp=nlp, meta=('getMaxlengthWord', 'object'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getMaxlengthWord, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "MaxlengthWord"
