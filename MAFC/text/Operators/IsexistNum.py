from spacy.lang.en import English
from spacy.tokens import Doc
from logger.logger import logger
from properties.properties import theproperty
from text.TextOperator import TextOperator


class IsexistNum(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getExistNum(data, nlp):
            doc = nlp(data)
            def get_has_number(doc):
                return any(token.like_num for token in doc)
            Doc.set_extension('has_number', getter=get_has_number, force=True)
            if doc._.has_number:
                return 1
            return 0

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getExistNum, nlp=nlp, meta=('getExistNum', 'int'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getExistNum, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "IsexistNum"
