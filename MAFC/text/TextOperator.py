from abc import abstractmethod


class TextOperator:
    def __init__(self):
        pass

    @abstractmethod
    def process(self, data):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        pass

    @abstractmethod
    def getName(self):
        pass



