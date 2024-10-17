import cv2
import pandas as pd
from properties.properties import theproperty
from image.ImageOperator import ImageOperator
import numpy as np
from scipy.stats import describe
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.feature import canny
from logger.logger import logger
import dask.dataframe as dd


class DefaultOperator(ImageOperator):

    def __init__(self):
        super(DefaultOperator, self).__init__(theproperty.image_path)
        #skew:偏度 kurt:峰度
        #S:rgb2gray技术
        #H:Hog技术
        self.columnslist = ['R_mean', 'G_mean', 'B_mean', 'R_var', 'G_var',
                            'B_var', 'R_kurt', 'G_kurt', 'B_kurt', 'R_skew',
                            'G_skew', 'B_skew', 'R_med', 'G_med', 'B_med',
                            'S_mean', 'S_var', 'S_kurt', 'S_skew', 'S_med',
                            'H_mean', 'H_var', 'H_kurt', 'H_skew', 'H_med', 'E_true', 'E_false']

    def process(self, imagedata: dd.DataFrame):
        if self.imagepath is None:
            logger.Error("the path of imagedata is not exist")
        if theproperty.dataframe == "dask":
            datalist = imagedata.iloc[:, -1].values.compute()
        elif theproperty.dataframe == "pandas":
            datalist = imagedata.iloc[:, -1].values
        stats = []
        for dlt in datalist:
            ipath = self.imagepath + dlt
            imagecv = cv2.imread(ipath)
            cv_stats = self.Dig_all_feature(imagecv)
            stats.append(cv_stats)
        data = pd.DataFrame(stats, columns=self.columnslist)
        if theproperty.dataframe == "dask":
            data = dd.from_pandas(data, npartitions=1)
        data.name = self.getName()
        #dataview = data.compute()
        return data

    def getName(self):
        return "DefaultOperator"

    def Dig_feature(self, image=None):
        ans = None
        if image is None:
            return ans
        return np.round(np.array(
            [np.mean(image), np.var(image), np.mean((image - np.mean(image)) ** 4) / pow(np.var(image), 2),
             np.mean((image - np.mean(image)) ** 3), np.median(image)]), 2)

    def Dig_all_feature(self, image=None):
        ans = None
        if image is None:
            return ans
        image_rgb = image.reshape((image.shape[0] * image.shape[1]), 3).T
        description = describe(image_rgb, axis=1)
        image_gs = rgb2gray(image)
        image_gs_feature = self.Dig_feature(image_gs)
        fd_image, image_hog = hog(image_gs, orientations=8, pixels_per_cell=(8, 8),
                                  cells_per_block=(3, 3), visualize=True)
        image_hogs = exposure.rescale_intensity(image_hog, in_range=(0, 0.04))
        image_hogs_feature = self.Dig_feature(image_hogs)
        image_edges = canny(image_gs, sigma=3)
        image_edges_nums = np.array(
            [np.sum(np.array(image_edges).flatten() == True), np.sum(np.array(image_edges).flatten() == False)])
        ans = np.append(np.array([np.round(description.mean, 2), np.round(description.variance, 2),
                                  np.round(description.kurtosis, 2), np.round(description.skewness, 2),
                                  np.round(np.median(image_rgb, axis=1), 2)]).flatten(), np.array([image_gs_feature,
                                                                                                   image_hogs_feature]).flatten())
        return np.append(ans, image_edges_nums)
