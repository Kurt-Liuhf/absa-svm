from abc import abstractmethod
from properties.properties import theproperty
import dask.dataframe as dd
class ImageOperator:
    def __init__(self, imagepath: str=theproperty.image_path):
        self.imagepath = imagepath

    @abstractmethod
    def process(self, imagedata: dd.DataFrame):
        '''

        :param data:dask.dataframe
        :return: dask.series
        '''
        pass

    @abstractmethod
    def getName(self):
        pass