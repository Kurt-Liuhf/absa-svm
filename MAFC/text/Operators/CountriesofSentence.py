from spacy.matcher import PhraseMatcher

from text.TextOperator import TextOperator
from spacy.lang.en import English
from logger.logger import logger
from properties.properties import theproperty
from text.country import Countries


class CountriesofSentence(TextOperator):
    def __init__(self):
        pass

    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        nlp = English()
        def getCountries(data, nlp):
            doc = nlp(data)
            country_matcher = PhraseMatcher(nlp.vocab)
            countries = Countries()
            patterns = list(nlp.pipe(countries.getCountries()))
            country_matcher.add('COUNTRY', None, *patterns)
            # Call the matcher on the test document and print the result
            matches = country_matcher(doc)
            return len(matches)

        if theproperty.dataframe == "dask":
            series = data.iloc[:, -1].apply(getCountries, nlp=nlp, meta=('getCountries', 'int'))
        elif theproperty.dataframe == "pandas":
            series = data.iloc[:, -1].apply(getCountries, nlp=nlp)
        else:
            logger.Info(f"{theproperty.dataframe} can not use !")

        return series

    def getName(self):
        return "CountriesofSentence"
