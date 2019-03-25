"""
Learning plotly Dash Layout

https://dash.plot.ly/getting-started
"""

import dash

# https://dash.plot.ly/dash-core-components
import dash_core_components as dcc

# https://dash.plot.ly/dash-html-components
import dash_html_components as html

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

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

dcc_opts = [
    {'label': 'New York City', 'value': 'NYC'},
    {'label': u'Montréal', 'value': 'MTL'},
    {'label': 'San Francisco', 'value': 'SF'},
]


df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/'
                 '5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/'
                 'gdp-life-exp-2007.csv')

app.layout = html.Div([
    html.Div(
        style={'backgroundColor': colors['background']}, children=[
        html.H1(children='Hello Dash World', style={'textAlign': 'center', 'color': colors['text']}),
        html.Div(children='''
            Dash: A web application framework for Python.
            Isn't that neat.
        ''', style={'textAlign': 'center', 'color': colors['text']}),
        dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': [1,2,3], 'y': [4,1,2], 'type': 'bar', 'name': 'SF'},
                    {'x': [1,2,3], 'y': [2,4,5], 'type': 'bar', 'name': u'Montréal'},
                ],
                'layout': {
                    'title': 'Dash Data Visualization',
                    'plot_bgcolor': colors['background'],
                    'paper_bgcolor': colors['background'],
                    'font': {
                        'color': colors['text']
                    }
                }
            }
        ),

        dcc.Graph(
            id='life-exp-vs-dgp',
            figure={
                'data': [
                    go.Scatter(
                        x=df.loc[df['continent'] == i, 'gdp per capita'],
                        y=df.loc[df['continent'] == i, 'life expectancy'],
                        text=df.loc[df['continent'] == i, 'country'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df['continent'].unique()
                ],
                'layout': go.Layout(
                    xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                    yaxis={'title': 'Life Expectancy'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
            }
        ),

        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Bar(
                        x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                           2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                        y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
                           350, 430, 474, 526, 488, 537, 500, 439],
                        name='Rest of world',
                        marker=go.bar.Marker(
                            color='rgb(55, 83, 109)'
                        )
                    ),
                    go.Bar(
                        x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
                           2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
                        y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,
                           299, 340, 403, 549, 499],
                        name='China',
                        marker=go.bar.Marker(
                            color='rgb(26, 118, 255)'
                        )
                    )
                ],
                layout=go.Layout(
                    title='US Export of Plastic Scrap',
                    showlegend=True,
                    legend=go.layout.Legend(
                        x=0,
                        y=1.0
                    ),
                    margin=go.layout.Margin(l=40, r=0, t=40, b=30)
                )
            ),
            style={'height': 300},
            id='my-graph'
        )
    ]),

    html.Div([
        html.Label('Dropdown'),
        dcc.Dropdown(
            options=dcc_opts,
            value='MTL'
        ),

        html.Label('Multi-Select Dropdown'),
        dcc.Dropdown(
            options=dcc_opts,
            value=['MTL', 'SF'],
            multi=True
        ),

        html.Label('Radio Items'),
        dcc.RadioItems(
            options=dcc_opts,
            value='MTL'
        ),

        html.Label('Checkboxes'),
        dcc.Checklist(
            options=dcc_opts,
            values=['MTL', 'SF']
        ),

        html.Label('Text Input'),
        dcc.Input(value='MTL', type='text'),

        html.Label('Slider'),
        dcc.Slider(
            min=0,
            max=9,
            marks={i: f'Label {i}' for i in range(1, 6)},
            value=5
        ),
    ], style={'columnCount': 2})

])


if __name__ == '__main__':
    app.run_server(debug=True)











