import dash  # pip install dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
from dash.dependencies import Output, Input
from collections import deque
from jupyter_dash import JupyterDash  # pip install jupyter-dash
from itertools import count

cc = count()
n = 20
X = deque(range(20), maxlen=n)
X.append(0)
Y = deque([0] * n, maxlen=n)
Y.append(0)

from hum.gen import signal_generators as sgen

# infinite generator of symbols and of outliers
symb_res = sgen.categorical_gen(sgen.gen_words)
out_res = sgen.bernoulli_gen(p_out=0.01)
# create an infinite signal from the 2 generators above
csigs = sgen.signal(symb_res, out_res, alphabet=list('abcde'))

app = JupyterDash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1 * 1000
        )
    ]
)


@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'n_intervals')])
def update_graph_scatter(input_data):
    X.append(next(cc))
    Y.append(next(csigs))

    data = plotly.graph_objs.Scatter(
        x=list(X),
        y=list(Y),
        name='Scatter',
        mode='lines+markers'
    )

    return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(X), max(X)]),
                                                yaxis=dict(range=[min(Y), max(Y)]), )}


if __name__ == '__main__':
    # run in a jupyter cell:
    app.run_server(mode='inline')
