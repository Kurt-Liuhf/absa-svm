from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class MinlengthWord(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getMinlengthWord(data, nlp):
            for i in [',', '.', '!', '?', '#', '%', ':', ';', '"', "-", '/', '_', "(", ")"]:
                data = data.replace(i, "")
            words = nlp(data)
            word = None
            for wd in words:
                if word == None:
                    word = wd
                    continue
                if len(wd) < len(word):
                    word = wd
            return word

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getMinlengthWord, nlp=nlp, meta=('getMinlengthWord', 'object'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getMinlengthWord, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "MinlengthWord"
