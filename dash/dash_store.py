"""
New Dash Core Component Store

https://dash.plot.ly/dash-core-components/store
"""


import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
import dash_table

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 10)

import plotly as py
from plotly.offline import plot
import plotly.graph_objs as go


print(f'dash version: {dash.__version__}')
print(f'plotly version: {py.__version__}')
print(f'__name__: {__name__}')

external_style = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


app = dash.Dash(__name__, external_stylesheets=external_style)


app.layout = html.Div([
    dcc.Store(id='mem-store'),
    dcc.Store(id='loc-store', storage_type='local'),
    dcc.Store(id='ses-store', storage_type='session'),

    html.Div([
        html.Button('click to store in mem', id='mem-button'),
        html.Button('click to store in loc', id='loc-button'),
        html.Button('click to store in ses', id='ses-button')
    ]),

    html.Div([
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th('memory clicks'),
                    html.Th('local clicks'),
                    html.Th('session clicks')
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(0, id='mem-clicks'),
                    html.Td(0, id='loc-clicks'),
                    html.Td(0, id='ses-clicks')
                ])
            ])
        ])
    ]),

    html.Br(),
    html.Div([
        dcc.Store(id='df-store'),
        dcc.Store(id='df-filt'),
    ]),

    html.Div([
        html.Button('get df', id='pull-df'),
        html.Br(),
        dcc.Dropdown(id='count-drop', multi=True),
        dash_table.DataTable(id='df-table')
    ])
])


# callbacks
# memory button store callbacks
for store in ('mem', 'loc', 'ses'):

    @app.callback(Output(f'{store}-store', 'data'),
                  [Input(f'{store}-button', 'n_clicks')],
                  [State(f'{store}-store', 'data')])
    def on_click(n_clicks, data):
        if not n_clicks:
            raise PreventUpdate
        data = data or {'clicks': 0}
        data['clicks'] += 1
        return data

    @app.callback(Output(f'{store}-clicks', 'children'),
                  [Input(f'{store}-store', 'modified_timestamp')],
                  [State(f'{store}-store', 'data')])
    def on_data(ts, data):
        if ts is None:
            raise PreventUpdate
        data = data or {}
        return data.get('clicks', 0)

# country callbacks
@app.callback([Output('df-store', 'data'),
               Output('count-drop', 'options'),
              Output('count-drop', 'value')],
              [Input('pull-df', 'n_clicks')])
def pull_df(n_clicks):
    if n_clicks is None:
        raise PreventUpdate

    df = pd.read_csv(
        'https://raw.githubusercontent.com/'
        'plotly/datasets/master/gapminderDataFiveYear.csv')
    return df.to_dict('rows'), [{'value': x, 'label': x} for x in df['country'].unique()], ['Canada', 'United States']

@app.callback(Output('df-filt', 'data'),
              [Input('count-drop', 'value')],
               [State('df-store', 'data')])
def update_df(value, data):
    if value is None:
        raise PreventUpdate

    print('value', value)
    df = pd.DataFrame(data)
    if value:
        filt_df = df[df['country'].isin(value)]
    else:
        filt_df = df
    return filt_df.to_dict('rows')

@app.callback([Output('df-table', 'columns'),
               Output('df-table', 'data')],
              [Input('df-filt', 'modified_timestamp')],
              [State('df-filt', 'data')])
def update_table(ts, data):
    if ts is None:
        raise PreventUpdate
    columns = [{'name': i, 'id': i} for i in data[0].keys()]
    return columns, data






if __name__ == '__main__':
    app.run_server(debug=True, port=8077, threaded=True)














