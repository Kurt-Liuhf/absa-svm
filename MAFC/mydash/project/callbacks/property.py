from dash.dependencies import Input, Output, State
from properties.properties import theproperty
from mydash.project.server import app


def setproperty(maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, backmodelpath):
    theproperty.maxevaluationattsperiter = maxevaluationattsperiter
    theproperty.DiscretizerBinsNumber = DiscretizerBinsNumber
    theproperty.finalchosenopspath = finalchosenopspath
    theproperty.backmodelpath = backmodelpath


def proprint():
    print(theproperty.maxevaluationattsperiter)
    print(theproperty.DiscretizerBinsNumber)
    print(theproperty.finalchosenopspath)
    print(theproperty.backmodelpath)


@app.callback(
    Output('setting', 'children'),
    Input('start', 'n_clicks'),
    # [State('maxevaluationattsperiter', 'value'),
    #  State('DiscretizerBinsNumber', 'value'),
    #  State('finalchosenopspath', 'value'),
    #  State('backmodelpath', 'value')],
    prevent_initial_call=True
)
def my_call(n_clicks):
    return "OK"
# def setting_property(n_clicks, maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, backmodelpath):
#     setproperty(maxevaluationattsperiter, DiscretizerBinsNumber, finalchosenopspath, backmodelpath)
#     proprint()
#     return "设置成功"


class test:
    pass
