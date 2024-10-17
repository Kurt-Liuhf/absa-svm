from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from server import app
from views.index import index_page
from views.property import property_page
from views.FC import FC_page
from views.dataview import dataview_page
from views.dataresult import dataresult_page
from Callbacks import Test
app.layout = html.Div(
    [
        # 监听url变化
        dcc.Location(id='url'),
        html.Div(
            [
                # 标题区域
                html.Div(
                    html.H3(
                        'MAFC',
                        style={
                            'marginTop': '20px',
                            'fontFamily': 'SimSun',
                            'fontWeight': 'bold'
                        }
                    ),
                    style={
                        'textAlign': 'center',
                        'margin': '0 10px 0 10px',
                        'borderBottom': '2px solid black'
                    }
                ),

                # 子页面区域
                html.Hr(),

                dbc.Nav(
                    [
                        dbc.NavLink('首页', href='/', active="exact", style={'backgroundColor': '#D1BBFF '}),
                        dbc.NavLink('参数配置', href='/property', active="exact", style={'backgroundColor': '#D1BBFF'}),
                        dbc.NavLink('特征构造', href='/FC', active="exact", style={'backgroundColor': '#D1BBFF'}),
                        dbc.NavLink('数据展示', href='/dataview', active="exact", style={'backgroundColor': '#D1BBFF'}),
                        dbc.NavLink('结果展示', href='/dataresult', active="exact", style={'backgroundColor': '#D1BBFF'}),
                    ],
                    vertical=True,
                    pills=True,
                )
            ],
            style={
                'flex': 'none',
                'width': '300px',
                'backgroundColor': '#CCEEFF'
            }
        ),
        html.Div(
            id='page-content',
            style={
                'flex': 'auto'
            }
        )
    ],
    style={
        'width': '100vw',
        'height': '100vh',
        'display': 'flex'
    }
)


# 路由总控
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return index_page

    elif pathname == '/property':
        return property_page

    elif pathname == '/FC':
        return FC_page

    elif pathname == '/dataview':
        return dataview_page

    elif pathname == '/dataresult':
        return dataresult_page

    return html.H1('您访问的页面不存在！')


if __name__ == '__main__':
    Test()
    app.run_server(debug=False)
