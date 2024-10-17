from dash import html, dcc, ClientsideFunction, Output, Input
#import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash import dash_table
import pandas as pd

from mydash.project.server import app
from properties.properties import theproperty


df = pd.DataFrame()
df.insert(loc=0, column='#', value=df.index)


dataview_page = html.Div(
    [
        html.Div(
            [
                dbc.Container(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.RadioItems(
                                    options=[
                                        {'label': 'filterdataview', 'value': '生成数据查看'},
                                        {'label': 'originaldataview', 'value': '原始数据'},
                                        {'label': 'traindatasetview', 'value': '训练集数据'},
                                        {'label': 'testdatasetview', 'value': '测试集数据'},
                                    ],
                                    id='datasetview-switch',
                                    value='原始数据',
                                    style={
                                        'position': 'fixed',
                                        'left': '320px',
                                        'top': '20px',
                                        'zIndex': '999'
                                    },
                                    inline=True
                                    ),
                                ),
                                dbc.Col(
                                    dbc.Label(id='dataview_out'),
                                    width=5,
                                    style={
                                        'position': 'fixed',
                                        'left': '950px',
                                        'top': '20px',
                                    },
                                ),
                            ]
                        )
                    ]
                ),
                dbc.Container(
                    dash_table.DataTable(
                        id='dataview_table',
                        columns=[{'name': column, 'id': column} for column in df.columns],
                        data=df.to_dict('records'),
                        virtualization=True,
                        sort_action='native',
                        style_table={
                            'height': '1000px',
                            'width': '800px',
                        },
                        style_cell={
                            'font-family': 'Times New Roman',
                            'text-align': 'center'
                        },
                        style_header_conditional=[
                            {
                                'if': {
                                    # 选定列id为#的列
                                    'column_id': '#'
                                },
                                'font-weight': 'bold',
                                'font-size': '24px'
                            }
                        ],
                        style_data_conditional=[
                            {
                                'if': {
                                    # 选中行下标为奇数的行
                                    'row_index': 'odd'
                                },
                                'background-color': '#cfd8dc'
                            }
                        ]
                    ),

                    style={
                        'margin-top': '100px',
                    }
                ),

            ],
        ),
        html.Div(
            [
                dbc.Col(
                    dbc.Label(id='feature_out'),
                    width=5,
                    style={
                        'position': 'fixed',
                        'left': '1150px',
                        'top': '40px',
                    },
                ),

                dcc.Graph(id="dataview-demo", style={"height": "90%", "width": "98%"}, config=dict(displayModeBar=False),)
            ], className="col-4 chart_div"
        ),

    ],
    style={
        'display': 'flex',
        'height': '100%'
    }
)

