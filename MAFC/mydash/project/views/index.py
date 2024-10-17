from dash import html
import dash_bootstrap_components as dbc
from mydash.project.server import app

index_page = html.Div(
    [
        html.P(
            [
                html.Span('欢迎来到基于'),
                html.Strong('plotly dash', style={'color': 'rgb(13, 103, 180)'}),
                html.Em(html.Sup('[1]', style={'fontSize': '16px'})),
                html.Span('编写的'),
                html.Strong('多模态自动特征构造系统  '),
                #html.Span('部分数据'),
                html.Em(html.Sup('[2]', style={'fontSize': '16px'})),
                #html.Span('简易看板示例，'),
                html.Span('作者：'),
                html.A('郑振亚', href='https://github.com/zzy728772024', target='_blank'),
                html.Span('。')
            ],
            id='index-desc',
            style={
                'margin': '50px 0 50px 50px'
            }
        ),
        html.Div(
            [
                dbc.Container(
                    [
                        html.P(
                            [
                                html.H3('Multimodal-Automated-Feature-Construction（MAFC）：', id='title'),
                                html.Span('该特征构造系统一共分为5个页面，首页是关于该系统的使用介绍；'),
                                html.A('参数配置页面', href='#title1'),
                                html.Span('是为了配置该系统所需的参数；'),
                                html.A('特征构造页面', href='#title2'),
                                html.Span('可以进行生成过滤器模型数据、设置数据集、特征构造、为测试集生成数据特征等操作；'),
                                html.A('数据展示页面', href='#title3'),
                                html.Span('可以展示经过特征构造后的数据集以及生成的过滤器模型数据；'),
                                html.A('结果展示页面', href='#title4'),
                                html.Span('是将特征构造过程中的指标数据用图标的形式展示出来，'
                                       '另外还展示了最终测试集的实验结果'),
                            ]
                        ),
                        html.Hr(),  # 水平分割线
                        html.H4('1：参数配置', id='title1'),
                        html.Hr(),  # 水平分割线
                        html.P('参数配置页面一共有14个参数输入框，将鼠标放在输入框就会有该参数的提示信息，所有参数输入完成后，点击最下方参数设置按钮即可。'),
                        html.Hr(),  # 水平分割线
                        html.H4('2：特征构造', id='title2'),
                        html.Hr(),  # 水平分割线
                        html.P('特征构造页面一共有四个功能：'),
                        html.P('（1）设置数据集：这是完成其它三个功能的前提，输入数据集路径参数，点击设置数据集按钮'),
                        html.P('（2）生成过滤器模型数据：该功能是从数据集中提取数据从而训练生成过滤器模型，此功能需要使用其它数据集，而非要进行特征构造的数据集'),
                        html.P('（3）特征构造：该功能是为数据集进行特征构造，如果选择迭代，则系统会进行特征评估和选择，否则只进行特征构造，另外将鼠标放在输入框能看到需要设置的参数的说明'),
                        html.P('（4）为测试集生成数据特征：要执行这个功能，需要先完成训练集的迭代特征构造操作，然后设置测试数据集路径，数据集名字要和训练集相同，最后点击为测试集生成数据特征按钮'),
                        html.P('其中特征构造自定义的扩展范式例子如下，假设文件名字为：mymafcoper.py，文件内容如下：'),
                        html.Code(''' 
import cv2
import pandas as pd
import dask.dataframe as dd
from MAFC_Operator.Unary import Unary
from text.TextOperator import TextOperator
from image.ImageOperator import ImageOperator
from MAFC_Operator.operator_base import outputType
from properties.properties import theproperty
from logger.logger import logger
from spacy.lang.en import English


class MyOperator(Unary):
"""
:param dataset:dask.dataframe.Dataframe or pandas.Dataframe
:param sourceColumns:{'name':列名,'type':outputType}
:param targetColumns:{'name':列名,'type':outputType}
    class outputType(Enum):
        Numeric = 0
        Discrete = 1
        Date = 2
        String = 3
:return: dask.core.series.Series or pandas.Series
"""
def __init__(self):
    super(MyOperator, self).__init__()

def processTrainingSet(self, dataset, sourceColumns, targetColumns):
    pass

def generateColumn(self, dataset, sourceColumns, targetColumns):
    columnname = sourceColumns[0]['name']

    def getweek(date):
        return date.weekday()

    if theproperty.dataframe == "dask":
        columndata = dataset[columnname].apply(getweek, meta=('getweek', 'int32'))
    elif theproperty.dataframe == "pandas":
        columndata = dataset[columnname].apply(getweek)
    else:
        logger.Info(f"no {theproperty.dataframe} can use")

    name = "DayofWeek(" + columnname + ")"
    newcolumn = {"name": name, "data": columndata}
    return newcolumn

def getName(self) -> str:
    return "MyOperator"

def isMatch(self, dataset, sourceColumns, targetColumns) -> bool:
    if super(MyOperator, self).isMatch(dataset, sourceColumns, targetColumns):
        if sourceColumns[0]['type'] == outputType.Date:
           return True
    return False


class MyTextOperator(TextOperator):
"""

"""
def __init__(self):
    super(MyTextOperator, self).__init__()

def process(self, data):
    nlp = English()
    def getCountWords(data, nlp):
        words = nlp(data)
        return len(words)

    if theproperty.dataframe == "dask":
        series = data.iloc[:, -1].apply(getCountWords, nlp=nlp, meta=('getCountWords', 'int'))
    elif theproperty.dataframe == "pandas":
        series = data.iloc[:, -1].apply(getCountWords, nlp=nlp)
    else:
        logger.Info(f"{theproperty.dataframe} can not use !")

    return series

def getName(self):
    return "MyTextOperator"

class MyImageOperator(ImageOperator):
"""

"""
def __init__(self):
    super(MyImageOperator, self).__init__(theproperty.image_path)
    self.columnslist = self.getName()

def process(self, imagedata):
    if self.imagepath is None:
        logger.Error("the path of imagedata is not exist")
    if theproperty.dataframe == "dask":
        datalist = imagedata.iloc[:, -1].values.compute()
    elif theproperty.dataframe == "pandas":
        datalist = imagedata.iloc[:, -1].values
    stats = []
    def withdrawfeature(image):
        return image.shape[0] + image.shape[1]
    for dlt in datalist:
        ipath = self.imagepath + dlt
        imagecv = cv2.imread(ipath)
        cv_stats = withdrawfeature(imagecv)
        stats.append(cv_stats)
    data = pd.DataFrame(stats, columns=self.columnslist)
    if theproperty.dataframe == "dask":
        data = dd.from_pandas(data, npartitions=1)
    data.name = self.getName()
    # dataview = data.compute()
    return data

def getName(self):
    return "MyImageOperator"
                            '''
                          , style={'language': 'python',
                            'white-space': 'pre',
                            'background-color': 'rgba(211, 211, 211, 0.25)',
                            }
                        ),

                        html.P('则需要将文件的绝对路径输入至“自定义的操作路径”参数框'),
                        html.Hr(),  # 水平分割线
                        html.H4('3：数据展示', id='title3'),
                        html.Hr(),  # 水平分割线
                        html.P('数据展示页面是为了展示经过特征构造前后的数据以及生成的过滤器模型数据，其中：'),
                        html.P('（1）filterdataview：为过滤器模型生成的数据'),
                        html.P('（2）originaldataview：原始数据融合后的数据'),
                        html.P('（3）traindatasetview：训练集特征构造后的数据'),
                        html.P('（4）testdatasetview：测试集特征构造后的数据'),

                        html.Hr(),  # 水平分割线
                        html.H4('4：结果展示', id='title4'),
                        html.P('结果展示分为训练集结果和测试集结果：'),
                        html.P('（1）训练集结果：特征构造过程中生成的新特征的过滤器得分：FScore和评估器得分：WSroce'),
                        html.P('（2）测试集结果：添加新特征前后测试集测试指标的对比'),
                        html.Hr(), # 水平分割线
                        html.A('回到顶端', href='#title'),
                    ]
                ),

            ],
            style={
                'display': 'flex',
                'justifyContent': 'left',
                'borderTop': '1px solid #e0e0e0',
                'borderBottom': '1px solid #e0e0e0',
                'margin': '0 50px 0 50px'
            }
        ),
        html.P(
            [
                html.Em('[1] '),
                html.Span('一个面向数据分析相关人员，纯Python即可实现交互式web应用，极大程度上简化web应用开发难度的先进框架',),
                html.Br(),
                html.Br(),
                html.Em('[2] '),
                html.Span('代码链接：'),
                html.A('https://github.com/zzy728772024/Multimodal-Automated-Feature-construction',
                       href='https://github.com/zzy728772024/Multimodal-Automated-Feature-construction',
                       target='_blank')
            ],
            style={
                'margin': '50px 0 0 50px'
            }
        )
    ]
)
