from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
#import dash_table
import pandas as pd
from properties.properties import theproperty



df = pd.DataFrame()
df.insert(loc=0, column='#', value=df.index)

dataresult_page = html.Div(
    [

        html.Div(
            [
                dbc.RadioItems(
                    options=[
                        {'label': '训练集结果', 'value': '训练集结果'},
                        {'label': '测试集结果', 'value': '测试集结果'}
                    ],
                    id='data-result-switch',
                    value='训练集结果',
                    style={
                        'position': 'fixed',
                        'right': '20px',
                        'top': '20px',
                        'zIndex': '999'
                    },
                    inline=True
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="iteration_id",
                        options=[
                            {'label': '第一次', 'value': 1},
                            {'label': '第二次', 'value': 2},
                            {'label': '第三次', 'value': 3},
                            {'label': '第四次', 'value': 4},
                        ],

                        placeholder='第一次',
                        value=1,
                    ),
                    width=8
                ),
                dcc.Graph(
                    id='dataresult-chart',
                    style={'height': '90%', 'weight': '900px'},
                    config={'displayModeBar': False},

                ),
            ],
            style={
                'flex': '1',
                'height': '100%'
            }
        )
    ],
    style={
        'display': 'flex',
        'height': '100%'
    }
)
