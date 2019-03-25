"""
Learning plotly Dash Callbacks

https://dash.plot.ly/getting-started
"""

import dash

# https://dash.plot.ly/dash-core-components
import dash_core_components as dcc

# https://dash.plot.ly/dash-html-components
import dash_html_components as html

from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 10)

import plotly as py
from plotly.offline import plot
import plotly.graph_objs as go

print(f'plotly version: {py.__version__}')
print(f'__name__: {__name__}')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    html.Div(id='my-div')
])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return f"you've entered {input_value}"


if __name__ == '__main__':
    app.run_server(debug=True)






