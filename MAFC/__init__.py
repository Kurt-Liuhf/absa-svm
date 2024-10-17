'''
欢迎使用MAFC(Multimo-Automated-Feature-construction)，这是一个多模态自动化特征构造工具，请根据以下步骤来使用它吧
import MAFC
1：help(MAFC.theproperty)，配置好相关参数
2：help(MAFC.Dataset)，创建数据集
3：help(generateModelData)，为数据集创建原特征数据
4：help(FC)进行特征构造
;
'''
import os, sys
root_path = os.path.dirname(__file__) # 父路径
sys.path.append(root_path)

from api import *
