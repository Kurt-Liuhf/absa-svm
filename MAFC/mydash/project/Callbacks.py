import copy
import os
from utils import execmyfunc
import dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dask.dataframe as dd
from logger.logger import logger
from properties.properties import theproperty
from server import app
from Dataset import Dataset
from utils import generateModelData
from FeatureConstruction import FC
from FeatureConstruction import generateTestData
import plotly.graph_objects as go


def setproperty(maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, maxFEvaluationnums,
                     backmodelpath, datasetlocation, resultfilepath, dataframe, temppath, wrapper, classifier, thread,
                     fsocre, wsocre):
    theproperty.maxevaluationattsperiter = maxevaluationattsperiter
    theproperty.DiscretizerBinsNumber = DiscretizerBinsNumber
    theproperty.finalchosenopspath = finalchosenopspath
    theproperty.maxFEvaluationnums = maxFEvaluationnums
    theproperty.backmodelpath = backmodelpath
    theproperty.datasetlocation = datasetlocation
    theproperty.resultfilepath = resultfilepath
    theproperty.temppath = temppath
    theproperty.dataframe = dataframe
    theproperty.wrapper = wrapper
    theproperty.classifier = classifier
    theproperty.thread = thread
    theproperty.fsocre = fsocre
    theproperty.wsocre = wsocre


@app.callback(
    [Output('maxevaluationattsperiter-local', 'data'),
     Output('DiscretizerBinsNumber-local', 'data'),
     Output('finalchosenopspath-local', 'data'),
     Output('maxFEvaluationnums-local', 'data'),
     Output('backmodelpath-local', 'data'),
     Output('datasetlocation-local', 'data'),
     Output('resultfilepath-local', 'data'),
     Output('temppath-local', 'data'),
     Output('dataframe-local', 'data'),
     Output('wrapper-local', 'data'),
     Output('classifier-local', 'data'),
     Output('thread-local', 'data'),
     Output('fsocre-local', 'data'),
     Output('wsocre-local', 'data'),],
    [Input('maxevaluationattsperiter', 'value'),
     Input('DiscretizerBinsNumber', 'value'),
     Input('finalchosenopspath', 'value'),
     Input('maxFEvaluationnums', 'value'),
     Input('backmodelpath', 'value'),
     Input('datasetlocation', 'value'),
     Input('resultfilepath', 'value'),
     Input('temppath', 'value'),
     Input('dataframe', 'value'),
     Input('wrapper', 'value'),
     Input('classifier', 'value'),
     Input('thread', 'value'),
     Input('fsocre', 'value'),
     Input('wsocre', 'value'),]
)
def FC_in_local_save_data(value0, value1, value2, value3, value4, value5, value6,
                          value7, value8, value9, value10, value11, value12, value13,):
    if value0 or value1 or value2 or value3 or value4 or value5 or value6\
            or value7 or value8 or value9 or value10 or value11 or value12 or value13:
        return value0, value1, value2, value3, value4, value5, value6,\
                value7, value8, value9, value10, value11, value12, value13,

    return dash.no_update


@app.callback(
    [Output('maxevaluationattsperiter', 'value'),
     Output('DiscretizerBinsNumber', 'value'),
     Output('finalchosenopspath', 'value'),
     Output('maxFEvaluationnums', 'value'),
     Output('backmodelpath', 'value'),
     Output('datasetlocation', 'value'),
     Output('resultfilepath', 'value'),
     Output('temppath', 'value'),
     Output('dataframe', 'value'),
     Output('wrapper', 'value'),
     Output('classifier', 'value'),
     Output('thread', 'value'),
     Output('fsocre', 'value'),
     Output('wsocre', 'value'),],
    [Input('maxevaluationattsperiter-local', 'data'),
     Input('DiscretizerBinsNumber-local', 'data'),
     Input('finalchosenopspath-local', 'data'),
     Input('maxFEvaluationnums-local', 'data'),
     Input('backmodelpath-local', 'data'),
     Input('datasetlocation-local', 'data'),
     Input('resultfilepath-local', 'data'),
     Input('temppath-local', 'data'),
     Input('dataframe-local', 'data'),
     Input('wrapper-local', 'data'),                 
     Input('classifier-local', 'data'),
     Input('thread-local', 'data'),
     Input('fsocre-local', 'data'),
     Input('wsocre-local', 'data'),]
)
def FC_in_local_placeholder(value0, value1, value2, value3, value4, value5, value6,
                          value7, value8, value9, value10, value11, value12, value13,):
    if value0 or value1 or value2 or value3 or value4 or value5 or value6\
            or value7 or value8 or value9 or value10 or value11 or value12 or value13:
        return value0, value1, value2, value3, value4, value5, value6,\
                value7, value8, value9, value10, value11, value12, value13,

    return dash.no_update


@app.callback(
    Output('Pro_setting_out', 'children'),
    Input('Pro_setting', 'n_clicks'),
    [State('maxevaluationattsperiter', 'value'),
     State('DiscretizerBinsNumber', 'value'),
     State('finalchosenopspath', 'value'),
     State('maxFEvaluationnums', 'value'),
     State('backmodelpath', 'value'),
     State('datasetlocation', 'value'),
     State('resultfilepath', 'value'),
     State('dataframe', 'value'),
     State('temppath', 'value'),
     State('wrapper', 'value'),
     State('classifier', 'value'),
     State('thread', 'value'),
     State('fsocre', 'value'),
     State('wsocre', 'value'),
     ],
    prevent_initial_call=True
)
def setting_property(n_clicks, maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, maxFEvaluationnums,
                     backmodelpath, datasetlocation, resultfilepath, dataframe, temppath, wrapper, classifier, thread,
                     fsocre, wsocre):
    setproperty(maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, maxFEvaluationnums,
                     backmodelpath, datasetlocation, resultfilepath, dataframe, temppath, wrapper, classifier, thread,
                     fsocre, wsocre)
    #proprint()
    return "参数设置成功"

data = None
dataset = None
settingflag = False


# local对应回调
@app.callback(
    [Output('dataset_name-local', 'data'),
     Output('image_path-local', 'data'),
     Output('text_path-local', 'data'),
     Output('tabular_path-local', 'data')],
    [Input('dataset_name', 'value'),
     Input('image_path', 'value'),
     Input('text_path', 'value'),
     Input('tabular_path', 'value'),]
)
def dataset_in_local_save_data(value0, value1, value2, value3):
    if value0 or value1 or value2 or value3:
        return value0, value1, value2, value3

    return dash.no_update


@app.callback(
    [Output('dataset_name', 'value'),
     Output('image_path', 'value'),
     Output('text_path', 'value'),
     Output('tabular_path', 'value')],
    [Input('dataset_name-local', 'data'),
     Input('image_path-local', 'data'),
     Input('text_path-local', 'data'),
     Input('tabular_path-local', 'data'), ]
)
def dataset_in_local_placeholder(value0, value1, value2, value3):
    if value0 or value1 or value2 or value3:
        return value0, value1, value2, value3

    return dash.no_update

@app.callback(
    [Output('is_iter-local', 'data'),
     Output('iternums-local', 'data'),
     Output('oper_path-local', 'data'),
     Output('image_oper-local', 'data'),
     Output('text_oper-local', 'data'),
     Output('oper_oper-local', 'data')],
    [Input('is_iter', 'value'),
     Input('iternums', 'value'),
     Input('oper_path', 'value'),
     Input('image_oper', 'value'),
     Input('text_oper', 'value'),
     Input('oper_oper', 'value')]
)
def FC_in_local_save_data(value0, value1, value2, value3, value4, value5):
    if value0 or value1 or value2 or value3 or value4 or value5:
        return value0, value1, value2, value3, value4, value5

    return dash.no_update


@app.callback(
    [Output('is_iter', 'value'),
     Output('iternums', 'value'),
     Output('oper_path', 'value'),
     Output('image_oper', 'value'),
     Output('text_oper', 'value'),
     Output('oper_oper', 'value')],
    [Input('is_iter-local', 'data'),
     Input('iternums-local', 'data'),
     Input('oper_path-local', 'data'),
     Input('image_oper-local', 'data'),
     Input('text_oper-local', 'data'),
     Input('oper_oper-local', 'data')]
)
def FC_in_local_placeholder(value0, value1, value2, value3, value4, value5):
    if value0 or value1 or value2 or value3 or value4 or value5:
        return value0, value1, value2, value3, value4, value5

    return dash.no_update


@app.callback(
    Output('FC_settingdata_out', 'children'),
    Input('FC_settingdata', 'n_clicks'),
    [State('dataset_name', 'value'),
     State('image_path', 'value'),
     State('text_path', 'value'),
     State('tabular_path', 'value'),
     ],
    prevent_initial_call=True
)
def setting_dataset(n_clicks, name, image_path, text_path, tabular_path):

    if n_clicks >= 1:
        if image_path == "":
            image_path = None
        else:
            image_path = theproperty.rootpath + image_path
        if text_path == "":
            text_path = None
        else:
            text_path = theproperty.rootpath + text_path
        tabular_path = theproperty.rootpath + tabular_path
        datapath = {"image_path": image_path, "text_path": text_path, "tabular_path": tabular_path}
        global dataset
        dataset = Dataset(datapath, name=name)
        global settingflag
        settingflag = True
        return "数据集设置成功"


@app.callback(
    Output('FC_GenerateData_out', 'children'),
    Input('FC_GenerateData', 'n_clicks'),
)
def mygeneratemodeldata(n_clicks):
    if n_clicks >= 1:
        global settingflag, dataset
        if settingflag is False:

            return "请先设置数据集"
        else:
            generateModelData(dataset)
            return "生成模型数据成功"

@app.callback(
    Output('FC_startFC', 'children'),
    Input('FC_strat', 'n_clicks'),
    [State('is_iter', 'value'),
     State('iternums', 'value'),
     State('oper_path', 'value'),
     State('image_oper', 'value'),
     State('text_oper', 'value'),
     State('oper_oper', 'value'),
     State('image_ign_oper', 'value'),
     State('text_ign_oper', 'value'),
     State('oper_ign_oper', 'value'),
     ],
    prevent_initial_call=True
)
def FC_dataset(n_clicks, is_iter, iternums, oper_path, image_oper, text_oper, oper_oper,
               image_ign_oper, text_ign_oper, oper_ign_oper):
    global dataset, settingflag
    if settingflag is False:
        return "数据集未设置"
    if n_clicks >= 1:
        if oper_path != "":
            if os.path.isfile(theproperty.rootpath + oper_path):
                theproperty.extendpath = oper_path
                #execmyfunc(codes)
        selfoper = {"image": None, "text": None, "tabular": None}
        ignoper = {"image": None, "text": None, "tabular": None}
        if image_oper != "":
            selfoper["image"] = eval(image_oper)
        if text_oper != "":
            selfoper["text"] = eval(text_oper)
        if oper_oper != "":
            selfoper["tabular"] = eval(oper_oper)
        if image_ign_oper != "":
            ignoper["image"] = eval(image_ign_oper)
        if text_ign_oper != "":
            ignoper["text"] = eval(text_ign_oper)
        if oper_ign_oper != "":
            ignoper["tabular"] = eval(oper_ign_oper)
        global data
        data = FC(dataset, is_iter == 1, iternums, selfoper, ignoper)
        return "特征构建完成"


@app.callback(
    Output('FC_TestData_out', 'children'),
    Input('FC_TestData', 'n_clicks'),
    prevent_initial_call=True
)
def FC_TestData(n_clicks):

    if n_clicks >= 1:
        global dataset, settingflag
        if settingflag is False:
            return "数据集未设置"
        generateTestData(dataset, dataset.name)
        return "测试集特征构建完成"

def getdf(value):
    try:
        df = None
        if value == "原始数据":
            df = (dd.read_csv(
                theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "original/*.part")
                  if theproperty.dataframe == "dask" else pd.read_csv(
                theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "original/" + theproperty.dataframe + theproperty.datasetname + "original.csv"))
        elif value == '生成数据查看':
            df = pd.read_csv(theproperty.rootpath + theproperty.datasetlocation + theproperty.datasetname + "_candidatedata.csv")

        elif value == "训练集数据":
            df = (dd.read_csv(
                    theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "/*.part")
                if theproperty.dataframe == "dask" else pd.read_csv(
                    theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "/" + theproperty.dataframe + theproperty.datasetname + ".csv"))

        else:
            df = (dd.read_csv(
                theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "test/*.part")
                  if theproperty.dataframe == "dask" else pd.read_csv(
                theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "test/" + theproperty.dataframe + theproperty.datasetname + "test.csv"))
        if theproperty.dataframe == "dask":
            df = df.compute()
    except Exception as ex:
        logger.Error(f"dataview_switch error: {ex}", ex)
    finally:
        return df
@app.callback(
    [Output('dataview_table', 'data'),
    Output('dataview_table', 'columns'),
     Output('dataview_out', 'children')
     ],
    Input('datasetview-switch', 'value'),
)
def dataview_switch(value):
    df = getdf(value)
    if df is None:
        return None, None, "数据不存在！"
    df.insert(loc=0, column='#', value=df.index)
    return df.to_dict('records'), [{'name': column, 'id': column} for column in df.columns], "数据加载完成！"

color_scale = [ '#8D7DFF','#2c0772', '#CDCCFF', '#ff2c6d']

@app.callback(
    [Output('dataview-demo', 'figure'),
    Output('feature_out', 'children')],
    Input('datasetview-switch', 'value'),
)
def get_pie(value):
    values = [0, 0, 0, 0]
    df = getdf(value)
    if df is not None:
        for index in df.columns:
            if df[index].dtype in ['float32', 'float64']:
                values[1] += 1
            elif df[index].dtype in ['int32', 'int64']:
                values[0] += 1
            elif df[index].dtype in ['datetime32[ns]', 'datetime64[ns]']:
                values[2] += 1
            else: values[3] += 1
    df_types = pd.DataFrame(values, index=['Discrete', 'Numeric', 'Date', 'String'], columns=['type'])
    trace = go.Pie(
        labels=df_types.index,
        values=df_types['type'],
        marker=dict(colors=color_scale[:len(df_types.index)])
    )
    layout = go.Layout(
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return go.Figure(data=[trace], layout=layout), f"共有{sum(values)}个特征，占比情况如下："

scty1 = [
        3607, 3834, 3904, 4080, 3997,
        3999, 3956, 4106, 4371, 4401,
    ]
scty2 = [
    4083, 3982, 3940, 3825, 3799,
    3935, 4187, 4037, 3844, 3862,
]


@app.callback(
    Output('dataresult-chart', 'figure'),
    [Input('data-result-switch', 'value'),
     Input('iteration_id', 'value'),]

)
def dataresult_switch_chart(value, iters):
    try:
        if value == '训练集结果':
            fig = go.Figure()
            df = pd.read_csv(theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "result.csv", index_col=False)
            namelist = str(df.loc[iters]["Added_Attributes"])
            LogLoss0 = float(df.loc[0]["LogLoss"])
            AUC0 = float(df.loc[0]["AUC"])
            F1Score0 = float(df.loc[0]["F1Score"])
            if len(namelist) > 2:
                namelist = namelist[1:-1].split('+')
            namelist2 = copy.deepcopy(namelist)
            namelist = []
            for nast in namelist2:
                newstr = ""
                index = 1
                for ss in nast:
                    newstr += ss
                    if index % 90 == 0:
                        newstr += '<br />'
                    index += 1
                namelist.append(newstr)
            print(len(namelist))
            scorenamelist = ['AUC', 'LogLoss', 'F1Score']
            LogLoss = float(df.loc[iters]["LogLoss"])
            AUC = float(df.loc[iters]["AUC"])
            F1Score = float(df.loc[iters]["F1Score"])
            scorelist0 = [AUC0, LogLoss0, F1Score0]
            scorelist1 = [AUC, LogLoss, F1Score]
            Chosen_Attributes_FScore = str(df.loc[iters]["Chosen_Attributes_FScore"])
            if len(Chosen_Attributes_FScore) > 2:
                Chosen_Attributes_FScore = Chosen_Attributes_FScore[1:-2].split(',')
                Chosen_Attributes_FScore = [float(i) for i in Chosen_Attributes_FScore]
            Chosen_Attributes_WScore = str(df.loc[iters]["Chosen_Attributes_WScore"])
            if len(Chosen_Attributes_WScore) > 2:
                Chosen_Attributes_WScore = Chosen_Attributes_WScore[1:-2].split(',')
                Chosen_Attributes_WScore = [float(i) for i in Chosen_Attributes_WScore]

            fig.add_trace(go.Bar(
                y=namelist,
                x=Chosen_Attributes_FScore,
                name='FScore',
                marker_color='indianred',
                text=Chosen_Attributes_FScore,
                textposition='outside',
                orientation='h'
            ))
            fig.add_trace(go.Bar(
                y=namelist,
                x=Chosen_Attributes_WScore,
                text=Chosen_Attributes_WScore,
                textposition='outside',
                name='WScore',
                marker_color='lightsalmon',
                orientation='h'
            ))

            fig.update_layout(
                font=dict(
                    family="Times New Roman, SimSun",
                    size=10,
                )
            )

            fig.update_layout(
                title_font_family="Times New Roman, SimSun",

            )

            fig.update_layout(
                margin=dict(t=10, b=10)
            )

        else:
            #fig = go.Figure()
            df = pd.read_csv(
                theproperty.rootpath + theproperty.resultfilepath + theproperty.dataframe + theproperty.datasetname + "testresult.csv", index_col=False)
            namelist = ['AUC', 'LogLoss', 'F1Score']
            LogLoss0 = float(df.loc[0]["LogLoss"])
            AUC0 = float(df.loc[0]["AUC"])
            F1Score0 = float(df.loc[0]["F1Score"])
            LogLoss1 = float(df.loc[1]["LogLoss"])
            AUC1 = float(df.loc[1]["AUC"])
            F1Score1 = float(df.loc[1]["F1Score"])
            scorelist0 = [AUC0, LogLoss0, F1Score0]
            scorelist1 = [AUC1, LogLoss1, F1Score1]

            scty1 = [
                AUC0, LogLoss0, F1Score0
            ]
            scty2 = [
                AUC1, LogLoss1, F1Score1
            ]
            trace1 = go.Bar(
                x=namelist,
                y=scorelist0,
                name='Original',
                text=scty1,
                textposition='outside',
            )
            trace2 = go.Bar(
                x=namelist,  # 横轴名称
                y=scorelist1,  # 纵轴数据
                name='After_FC',  # 右边小标签名称和浮标名称

                text=scty2,  # 设置浮标内容，可以是个数组，用于显示每个条目的说明或者需要一直显示的数据，配合textposition使用

                textposition='outside',

            )
            '''
                            insidetextanchor='start',  # 只有当textposition全部为inside之后才起作用，默认是‘end’。‘start’、‘middle’、‘end’

                            textfont={  # 设置text的样式
                                'family': 'Courier New',  # 设置字体
                                'size': 16,  # 字体大小
                                'color': '#000'  # 字体颜色
                            },

                            hovertext='After_FC',  # 设置浮标里面的内容

                            marker={  # 设置柱形图样式
                                'line': {  # 设置线的样式
                                    'width': 4,  # 线的宽度
                                    'color': '#00FF00',  # 先的颜色
                                },
                                'colorbar': {  # 设置颜色标尺，可以看出程度
                                    'thicknessmode': 'pixels',  # 设置标尺宽度模式，‘fraction’ 分数占比、‘pixels’ 像素值
                                    'thickness': 12,  # 设置具体占比和像素大小，标尺的宽度，如果是像素，则使用整数
                                    'lenmode': 'fraction',  # 标尺高度模式，‘fraction’ 分数占比、'pixels' 像素值
                                    'len': 0.8,  # 标尺高度具体数值，如果是像素，则使用整数
                                    'x': -0.08,  # 设置标尺位置，[-2, 3]之间
                                    'xanchor': 'center',  # 标尺的对齐方式，‘left’、‘center’、‘right’，默认‘left’
                                    'y': 0,  # 标尺的y位置
                                    'yanchor': 'bottom',  # 标尺y轴对齐方式，‘top’、‘middle’、‘bottom’，默认‘middle’
                                    'tickformat': '.2s',  # 设置标尺文字的匹配格式，比如百分比'.2%', 带k的整数'.2s'
                                    'title': {  # 标尺的内容，可以只是一个字符串，也可以像现在这样设置一个字典
                                        'text': '程度',  # 标尺的名称
                                        'font': {
                                            'size': 16,  # 大小
                                            'color': '#5982AD',  # 颜色
                                        },
                                        'side': 'top',  # 位置，‘right’、‘top’、‘bottom’，默认是‘top’
                                    },
                                },
                            },
                            '''
            layout = go.Layout(
                title='指标对比',
                barmode='group',  # 可以分为 ‘stack’(叠加）、‘group’（分组）、‘overlay’（重叠）、‘relative’（相关）， 默认是‘group’
                barnorm='',  # 设置柱形图纵轴或横轴数据的表示形式，可以是fraction（分数），percent（百分数）

            )
            fig = go.Figure(
                data=[trace1, trace2],
                layout=layout
            )



    except Exception as ex:
        logger.Error(f"dataresult_switch_chart error: {ex}", ex)
    finally:
        return fig


class Test:
    pass