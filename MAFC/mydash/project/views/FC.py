from dash import html, dcc

from mydash.project.callbacks.FC import test
import dash_bootstrap_components as dbc



test()
FC_page = html.Div(
    dbc.Container(
        [
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("数据集名称", addon_type="prepend"),
                            dbc.Input(
                                id='dataset_name',
                                #placeholder='5',
                                type="str",
                                value="MutilModaldata"
                            ),
                            dbc.InputGroupAddon("图片路径", addon_type="prepend"),
                            dbc.Input(
                                id='image_path',
                                #placeholder='5',
                                type="str",
                                value="data/datasettrain/image/"
                            ),
                            dbc.Tooltip('为数据集取个名字吧。',
                                target='dataset_name'),
                            dbc.Tooltip('图片数据路径，没有的话留空。',
                                        target='image_path'),
                            dcc.Store(id='dataset_name-local', storage_type='session'),
                            dcc.Store(id='image_path-local', storage_type='session'),

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
                            dbc.InputGroupAddon("文本路径", addon_type="prepend"),
                            dbc.Input(
                                id='text_path',
                                # placeholder='5',
                                type="str",
                                value="data/datasettrain/text/text.csv"
                            ),
                            dbc.InputGroupAddon("表格路径", addon_type="prepend"),
                            dbc.Input(
                                id='tabular_path',
                                # placeholder='5',
                                type="str",
                                value="data/datasettrain/tabular/data.csv"
                            ),
                            dbc.Tooltip('文本数据路径，没有的话留空。',
                                        target='text_path'),
                            dbc.Tooltip('表格数据路径，没有的话留空。',
                                        target='tabular_path'),
                            dcc.Store(id='text_path-local', storage_type='session'),
                            dcc.Store(id='tabular_path-local', storage_type='session'),
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button('设置数据集', id='FC_settingdata', n_clicks=0, color='light'),
                            dbc.Col(
                                dbc.Spinner(dbc.Label(id='FC_settingdata_out')),
                                width=5
                            ),

                            dbc.Tooltip('点击生成dataset。',
                                        target='FC_settingdata'),
                        ],
                    ),
                    width={'size': 6, 'offset': 3},
                ),
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button('生成过滤器模型数据', id='FC_GenerateData', n_clicks=0, color='light'),
                            dbc.Col(
                                dbc.Spinner(dbc.Label(id='FC_GenerateData_out')),
                                width=5
                            ),

                            dbc.Tooltip('开始从数据集中生成原特征数据。',
                                        target='FC_GenerateData')
                        ],
                    ),
                    width={'size': 6, 'offset': 3},
                ),
            ),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.InputGroupAddon("是否迭代", addon_type="prepend"),
                            dbc.Input(
                                id='is_iter',
                                placeholder='1',
                                type="number",
                                value=1
                            ),
                            dbc.InputGroupAddon("迭代次数", addon_type="prepend"),
                            dbc.Input(
                                id='iternums',
                                placeholder='2',
                                type="number",
                                value=4
                            ),
                            dbc.Tooltip('是否进行迭代操作，是1，否0。',
                                        target='is_iter'),
                            dbc.Tooltip('请输入迭代次数。',
                                        target='iternums'),
                            dcc.Store(id='is_iter-local', storage_type='session'),
                            dcc.Store(id='iternums-local', storage_type='session'),
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
                            dbc.InputGroupAddon("自定义的操作路径", addon_type="prepend"),
                            dbc.Input(
                                id='oper_path',
                                # placeholder='5',
                                type="str",
                                value="test/mymafcoper.py"
                            ),
                            dbc.InputGroupAddon("image", addon_type="prepend"),
                            dbc.Input(
                                id='image_oper',
                                # placeholder='5',
                                type="str",
                                value="['MyImageOperator']"
                            ),
                            dbc.Tooltip('自定义的操作代码文件路径，没有的话留空。',
                                        target='oper_path'),
                            dbc.Tooltip('自定义image操作名字，格式如下["name1","name2"...]，没有的话留空。',
                                        target='image_oper'),
                            dcc.Store(id='oper_path-local', storage_type='session'),
                            dcc.Store(id='image_oper-local', storage_type='session'),
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
                            dbc.InputGroupAddon("text", addon_type="prepend"),
                            dbc.Input(
                                id='text_oper',
                                # placeholder='5',
                                type="str",
                                value="['MyTextOperator']"
                            ),
                            dbc.InputGroupAddon("operator", addon_type="prepend"),
                            dbc.Input(
                                id='oper_oper',
                                # placeholder='5',
                                type="str",
                                value="{'unary':['MyOperator']}"
                            ),
                            dbc.Tooltip('自定义text操作名字，格式如下["name1","name2"...]，没有的话留空。',
                                        target='text_oper'),
                            dbc.Tooltip('自定义operator操作名字，格式如下{"unary":["name1","name2"...],"other":"["name1","name2"...]"}，没有的话留空。',
                                        target='oper_oper'),
                            dcc.Store(id='text_oper-local', storage_type='session'),
                            dcc.Store(id='oper_oper-local', storage_type='session'),
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
                            dbc.InputGroupAddon("忽略的操作", addon_type="prepend"),
                            dbc.InputGroupAddon("image", addon_type="prepend"),
                            dbc.Input(
                                id='image_ign_oper',
                                # placeholder='5',
                                type="str",
                                value=""
                            ),
                            dbc.InputGroupAddon("text", addon_type="prepend"),
                            dbc.Input(
                                id='text_ign_oper',
                                # placeholder='5',
                                type="str",
                                value="['CountWords']"
                            ),
                            dbc.InputGroupAddon("operator", addon_type="prepend"),
                            dbc.Input(
                                id='oper_ign_oper',
                                # placeholder='5',
                                type="str",
                                value="{'unary':['Discretizer']}"
                            ),
                            dbc.Tooltip('忽略的image操作名字，格式如下["name1","name2"...]，没有的话留空。',
                                        target='image_ign_oper'),
                            dbc.Tooltip('忽略的text操作名字，格式如下["name1","name2"...]，没有的话留空。',
                                        target='text_ign_oper'),
                            dbc.Tooltip('忽略的operator操作名字，格式如下{"unary":["name1","name2"...],"other":"["name1","name2"...]"}，没有的话留空。',
                                        target='oper_ign_oper'),
                            # dcc.Store(id='maxevaluationattsperiter-local', storage_type='local')
                        ],
                    ),
                    width={'size': 6, 'offset': 3}
                )
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button('开始特征构造', id='FC_strat', n_clicks=0, color='light'),
                            dbc.Col(
                                dbc.Spinner(dbc.Label(id='FC_startFC')),
                                width=4
                            ),

                            dbc.Tooltip('鼠标左键单击，开始愉快的进行特征构造吧。',
                                        target='FC_strat'),
                        ],
                    ),
                    width={'size': 6, 'offset': 3},
                ),
            ),
            html.Br(),
            html.Br(),
            dbc.Row(
                dbc.Col(
                    dbc.InputGroup(
                        [
                            dbc.Button('测试集数据特征构造', id='FC_TestData', n_clicks=0, color='light'),
                            dbc.Col(
                                dbc.Spinner(dbc.Label(id='FC_TestData_out')),
                                width=4
                            ),
                            dbc.Tooltip('该按钮是为测试数据集准备的，再保存好构造范式后再点击此按钮。',
                                        target='FC_TestData')
                        ],
                    ),
                    width={'size': 6, 'offset': 3},
                ),
            ),
        ],
    fluid=True
    )
)
