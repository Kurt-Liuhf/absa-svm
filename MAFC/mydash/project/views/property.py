from dash import html
from dash import dcc
import dash_bootstrap_components as dbc


property_page = html.Div(
    dbc.Container(
        [
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("maxevaluationattsperiter", addon_type="prepend"),
                            dbc.Input(
                                id='maxevaluationattsperiter',
                                placeholder='5',
                                type="number",
                                value=1500
                            ),
                            dbc.Tooltip('该参数是限制评估器最多评估多少个新特征。',
                                target='maxevaluationattsperiter'),
                            dcc.Store(id='maxevaluationattsperiter-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("DiscretizerBinsNumber", addon_type="prepend"),
                            dbc.Input(
                                id='DiscretizerBinsNumber',
                                placeholder='2',
                                type="number",
                                value=10
                            ),
                            dbc.Tooltip('该参数是规定数值型特征离散化后的个数。',
                                        target='DiscretizerBinsNumber'),
                            dcc.Store(id='DiscretizerBinsNumber-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("finalchosenopspath", addon_type="prepend"),
                            dbc.Input(
                                id='finalchosenopspath',

                                type="str",
                                value="data/finalchosen/"
                            ),
                            dbc.Tooltip('该参数是规定最终范式的保存路径。',
                                        target='finalchosenopspath'),
                            dcc.Store(id='finalchosenopspath-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("maxFEvaluationnums", addon_type="prepend"),
                            dbc.Input(
                                id='maxFEvaluationnums',

                                type="number",
                                value=3000
                            ),
                            dbc.Tooltip('该参数是限制过滤器最多评估多少个新特征。',
                                        target='maxFEvaluationnums'),
                            dcc.Store(id='maxFEvaluationnums-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("backmodelpath", addon_type="prepend"),
                            dbc.Input(
                                id='backmodelpath',
                                #placeholder='2',
                                type="str",
                                value="data/model/"
                            ),
                            dbc.Tooltip('该参数是规定过滤器模型的保存路径。',
                                        target='backmodelpath'),
                            dcc.Store(id='backmodelpath-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("datasetlocation", addon_type="prepend"),
                            dbc.Input(
                                id='datasetlocation',
                                placeholder='2',
                                type="str",
                                value="data/datasets/"
                            ),
                            dbc.Tooltip('该参数是规定过滤器模型需要的训练数据集存放的路径。',
                                        target='datasetlocation'),
                            dcc.Store(id='datasetlocation-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("resultfilepath", addon_type="prepend"),
                            dbc.Input(
                                id='resultfilepath',
                                #placeholder='2',
                                type="str",
                                value="data/result/"
                            ),
                            dbc.Tooltip('该参数是规定最终结果文件存放的路径。',
                                        target='resultfilepath'),
                            dcc.Store(id='resultfilepath-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("dataframe", addon_type="prepend"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='dataframe',
                                    options=[
                                        {'label': 'pandas', 'value': 'pandas'},
                                        {'label': 'dask', 'value': 'dask'},
                                    ],
                                    placeholder='pandas',
                                    value='pandas'
                                ),
                                width=8
                            ),

                            dbc.Label
                                (
                                id="mydataframe",
                                children="参数说明",
                            ),
                            dbc.Tooltip('该参数是选择使用数据处理工具。',
                                        target='mydataframe'),
                            dcc.Store(id='dataframe-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("temppath", addon_type="prepend"),
                            dbc.Input(
                                id='temppath',
                                #placeholder='2',
                                type="str",
                                value="data/temp/"
                            ),
                            dbc.Tooltip('该参数是规定中间文件存放的路径。',
                                        target='temppath'),
                            dcc.Store(id='temppath-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("wrapper", addon_type="prepend"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id='wrapper',
                                    options=[
                                        {'label': 'AucWrapperEvaluation', 'value': 'AucWrapperEvaluation'},
                                        {'label': 'LogLossEvaluation', 'value': 'LogLossEvaluation'},
                                        {'label': 'FoneEvaluation', 'value': 'FoneEvaluation'},
                                    ],
                                    placeholder='AucWrapperEvaluation',
                                    value='AucWrapperEvaluation',
                                ),
                                width=8
                            ),

                            dbc.Label
                            (
                                id="mywrapper",
                                children="参数说明",
                            ),
                            dbc.Tooltip('该参数是用到的评估器。',
                                        target='mywrapper'),
                            dcc.Store(id='wrapper-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("classifier", addon_type="prepend"),
                            dbc.Col(
                                dcc.Dropdown(
                                    id="classifier",
                                    options=[
                                        {'label': 'SVM', 'value': 'SVM'},
                                        {'label': 'DicisionTree', 'value': 'DicisionTree'},
                                        {'label': 'RandomForest', 'value': 'RandomForest'},
                                    ],
                                    placeholder='RandomForest',
                                    value='RandomForest',
                                ),
                                width=8
                            ),
                            dbc.Label
                            (
                                id="myclassifier",
                                children="参数说明"
                            ),
                            # dbc.Input(
                            #     id='classifier',
                            #     #placeholder='2',
                            #     type='str',
                            #     value='RandomForest'
                            # ),
                            dbc.Tooltip('该参数是选择用到的模型。',
                                        target="myclassifier"),
                            dcc.Store(id='classifier-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("thread", addon_type="prepend"),
                            dbc.Input(
                                id='thread',
                                placeholder='2',
                                type="number",
                                value=8
                            ),
                            dbc.Tooltip('该参数是选择采用多少线程进行运行。',
                                        target="thread"),
                            dcc.Store(id='thread-local', storage_type='session')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("fsocre", addon_type="prepend"),
                            dbc.Input(
                                id='fsocre',
                                placeholder='2',
                                type="number",
                                value=0.1
                            ),
                            dbc.Tooltip('该参数是选择过滤器的阈值。',
                                        target="fsocre"),
                            dcc.Store(id='fsocre-local', storage_type='session')
                        ],
                    ),

                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("wsocre", addon_type="prepend"),
                            dbc.Input(
                                id='wsocre',
                                #placeholder='2',
                                type="number",
                                value=0.01
                            ),
                            dbc.Tooltip('该参数是选择评估器的阈值。',
                                        target="wsocre"),
                            dcc.Store(id='wsocre-local', storage_type='session')
                        ],
                    ),

                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button('参数设置', id='Pro_setting', n_clicks=0, color='light'),
                            dbc.Col(
                                dbc.Label(id='Pro_setting_out'),
                                width=4
                            )

                        ],
                    ),
                    width={'size': 6, 'offset': 3},
                ),
            ),


        ],
    fluid=True
    )
)



