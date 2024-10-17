import dash

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True
)

# 设置网页title
app.title = 'MAFC'

server = app.server