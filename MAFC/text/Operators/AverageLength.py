from text.TextOperator import TextOperator
from spacy.lang.en import English
from properties.properties import theproperty
from logger.logger import logger


class AverageLength(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getAvglength(data, nlp):
            nums = 0
            words = nlp(data)
            for word in words:
                nums += len(word)
            return nums // len(words)
        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getAvglength, nlp=nlp, meta=('getAvglength', 'f8'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getAvglength, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")
        return series

    def getName(self):
        return "AverageLength"
