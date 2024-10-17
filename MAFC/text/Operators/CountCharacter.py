from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class CountCharacter(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getCountChara(data, nlp):
            nums = 0
            words = nlp(data)
            for word in words:
                nums += len(word)
            return nums

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getCountChara, nlp=nlp, meta=('getCountChara', 'int32'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getCountChara, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "CountCharacter"
