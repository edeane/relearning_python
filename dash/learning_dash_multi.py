"""
Learning plotly Dash Callbacks

https://dash.plot.ly/urls
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import numpy as np
import pandas as pd
import json
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 10)
np.random.seed(0)

import plotly as py
from plotly.offline import plot
import plotly.graph_objs as go

print(f'plotly version: {py.__version__}')
print(f'__name__: {__name__}')
print(dcc.__version__) # 0.6.0 or above is required

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# ---- app ----

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('go to page 2', href='/page-2')
])

page_1_layout = html.Div([
    html.H1('page 1'),
    dcc.Dropdown(id='page-1-dropdown', options=[{'label': i, 'value': i} for i in range(5)], value=0),
    html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

@app.callback(
    Output('page-1-content', 'children'),
    [Input('page-1-dropdown', 'value')]
)
def page_1_dropdown(value):
    return f'the selected value on page 1 is {value}'


page_2_layout = html.Div([
    html.H1('page 2'),
    dcc.RadioItems(id='page-2-radio', options=[{'label': i, 'value': i**2} for i in range(5)], value=0),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

@app.callback(
    Output('page-2-content', 'children'),
    [Input('page-2-radio', 'value')]
)
def page_2_radio(value):
    return f'this is the page 2 radio value: {value}'


@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    path_dict = {
        '/page-1': page_1_layout,
        '/page-2': page_2_layout,
    }
    return path_dict.get(pathname, index_page)

if __name__ == '__main__':
    app.run_server(debug=True)





