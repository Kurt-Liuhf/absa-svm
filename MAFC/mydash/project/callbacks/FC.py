from dash.dependencies import Input, Output, State
from properties.properties import theproperty
from mydash.project.server import app




@app.callback(
    Output('setting2', 'children'),
    Input('start2', 'n_clicks'),
    # [State('maxevaluationattsperiter', 'value'),
    #  State('DiscretizerBinsNumber', 'value'),
    #  State('finalchosenopspath', 'value'),
    #  State('backmodelpath', 'value')],
    prevent_initial_call=True
)
def my_call(n_clicks):
    return "OK"

class test:
    pass