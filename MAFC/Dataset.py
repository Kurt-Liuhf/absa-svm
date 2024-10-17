import dask.dataframe as dd
import os
import pandas as pd
from logger.logger import logger
from properties.properties import theproperty

class Dataset:
    def __init__(self, data_path=None, mode: int = 1, sql=None, name=None):
        '''
        :param data_path:数据集路径，是一个字典:{'image_path':"",'text_path':"",'tabular_path':""}
            图片的路径下需要给出match.csv文件，将读取图片的名字按顺序写入，如果不存在match.csv文件，则将按名称读取图片,例如match.csv内容为：
            id,image_data
            0,1.png
            1,2.png
            ...
            id要从零开始
            另外在图片的路径下新建datas文件夹，将图片存放到该文件夹，例如
            data_path = {"image_path":"/home/zzy/data/image/","text_path":"/home/zzy/data/text/text,csv","tabular":"/home/zzy/data/tabular/test.csv"}
            文件目录则如下：
            --/home/zzy/data
                --image
                    --datas
                        1.png
                        2.png
                    match.csv
                --text
                    text.csv
                --tabuluar
                    test.csv
            请注意文件目录最后的/，如果是目录需要以/结尾
        :param mode:数据的读取方式，1代表文件读取，2代表是数据库读取，默认为1
        :param sql:（暂不支持）数据库相关信息{"ip":"","port":"","user":"","password","","database":"","table":{"image_data":"","text_data":"","tabular_data":""}}
        :param name:数据集的名字
        '''
        self.mode = mode
        self.data_path = data_path
        self.data_tabular = self.read_tabular()
        self.data_text = self.read_text()
        self.data_image = self.read_image()
        self.name = name
        if name is not None:
            theproperty.datasetname = name

    def read_text(self):
        '''
        :return:Dask.Dataframe
        '''
        if self.data_path is None or self.data_path.get('text_path') is None:
            return None
        text_path = self.data_path['text_path']
        if text_path is None:
            return None
        if theproperty.dataframe == "dask":
            text_data = dd.read_csv(text_path)
        elif theproperty.dataframe == "pandas":
            text_data = pd.read_csv(text_path, sep='\n')
        else:
            logger.Info("no " + theproperty.dataframe + " can use!")

        return text_data


    def read_image(self):
        '''
        :return:Dask.Dataframe，有2个Series，存放的是序号和图片的名称
        '''
        if self.data_path is None or self.data_path.get('image_path') is None:
            return None
        image_path = self.data_path['image_path']
        if image_path is None:
            return None

        match_path = image_path + "match.csv"
        imagedata_path = image_path + "datas/"
        theproperty.image_path = imagedata_path
        image_data = None
        if os.path.isfile(match_path):
            if theproperty.dataframe == "dask":
                image_data = dd.read_csv(match_path)
            elif theproperty.dataframe == "pandas":
                image_data = pd.read_csv(match_path)
            else:
                logger.Info("no " + theproperty.dataframe + " can use!")

        else:
            if os.path.isdir(imagedata_path):
                imagelist = os.listdir(imagedata_path)
                if theproperty.dataframe == "dask":

                    theseries = pd.Series(imagelist, name="image_data", dtype="object")
                    image_data = dd.from_pandas(theseries, npartitions=1).reset_index()
                elif theproperty.dataframe == "pandas":
                    image_data = pd.DataFrame(imagelist, index=["image_data"])
                else:
                    logger.Info("no " + theproperty.dataframe + " can use!")
            else:
                logger.Error(imagedata_path + " is not exist!")
        return image_data

    def read_tabular(self):
        '''
        :return:Dask.Dataframe
        '''
        if self.data_path is None or self.data_path.get('tabular_path') is None:
            return None
        tabular_path = self.data_path['tabular_path']
        if tabular_path is None:
            return None
        if theproperty.dataframe == "dask":
            tabular_data = dd.read_csv(tabular_path)
        elif theproperty.dataframe == "pandas":
            tabular_data = pd.read_csv(tabular_path)
        else:
            logger.Info("no " + theproperty.dataframe + " can use!")

        return tabular_data

    def fromArray(self, data):
        if 'sparse' in str(type(data)):
            data = data.toarray()
        if len(data) > 0:
            self.data_tabular = pd.DataFrame(data, columns=[str(i) for i in range(len(data[0]))])