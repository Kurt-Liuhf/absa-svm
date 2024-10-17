from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty


class MaxNumberPreDollor(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getMaxDollor(data, nlp):
            doc = nlp(data)
            maxnums = 0
            for token in doc:
                if token.like_num:
                    if token.i == len(doc) - 1:
                        continue
                    # Get the next token in the document
                    next_token = doc[token.i + 1]
                    # Check if the next token's text equals '$'
                    if next_token.text == '$':
                        num = (int)(token.text)
                        if num > maxnums:
                            maxnums = num
            return maxnums

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getMaxDollor, nlp=nlp, meta=('getMaxDollor', 'f8'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getMaxDollor, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "MaxNumberPreDollor"
