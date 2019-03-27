"""
Learning plotly Dash Callbacks

https://dash.plot.ly/getting-started-part-2
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

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
df2 = pd.read_csv('https://gist.githubusercontent.com/chriddyp/cb5392c35661370d95f300086accea51/raw/'
                  '8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/indicators.csv')
df2.drop(df2.columns[0], axis=1, inplace=True)
avail_indi = df2['Indicator Name'].unique()
# x = 5
# exp_df = pd.DataFrame({'type': ['x ** 2', 'x ** 3', '2 ** x', '3 ** x', 'x ** x'],
#                        'value': [x ** 2, x ** 3, 2 ** x, 3 ** x, x ** x]})
all_options = {'America': ['NYC', 'SF', 'CI'],
               'Canada': ['Mont', 'Toro', 'Otta']}



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    dcc.Input(id='my-id', value='initial value', type='text'),
    html.Div(id='my-div'),
    dcc.Graph(id='graph-w-slider'),
    dcc.Slider(
        id='year-slider',
        min=df['year'].min(),
        max=df['year'].max(),
        value=df['year'].min(),
        marks={str(year): str(year) for year in df['year'].unique()}
    ),
    # entire thing
    html.Div([
        # first drop downs
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in avail_indi],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        # second drop downs
        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in avail_indi],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
    # graph
    dcc.Graph(id='indicator-graphic'),
    dcc.Slider(
        id='year--slider',
        min=df2['Year'].min(),
        max=df2['Year'].max(),
        value=df2['Year'].max(),
        marks={str(year): str(year) for year in df2['Year'].unique()}
    ),
    html.Hr(),
    dcc.Input('in-exp', 5, type='number'),
    html.Div(id='exp-tbl'),
    html.Div([
        dcc.RadioItems(
            id='countries-dropdown',
            options=[{'label': k, 'value': k} for k in all_options.keys()],
            value='America'
        ),
        dcc.RadioItems(id='cities-dropdown'),
        html.Div(id='display-selected-values')
    ])

])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return f"you've entered {input_value}"


@app.callback(
    Output('graph-w-slider', 'figure'),
    [Input('year-slider', 'value')]
)
def update_figure(selected_year):
    filtered_df = df[df['year'] == selected_year]
    traces = []
    for i in filtered_df['continent'].unique():
        df_by_continent = filtered_df[filtered_df['continent'] == i]
        traces.append(go.Scatter(
            x=df_by_continent['gdpPercap'],
            y=df_by_continent['lifeExp'],
            text=df_by_continent['country'],
            mode='markers',
            opacity=.7,
            marker={
                'size': 15,
                'line': {'width': .5, 'color': 'white'}
            },
            name=i
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'type': 'log', 'title': 'GDP Per Capita'},
            yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value'),
     Input('year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df2[df2['Year'] == year_value]
    print('Update Made')

    return {
        'data': [go.Scatter(
            x=dff.loc[dff['Indicator Name'] == xaxis_column_name, 'Value'],
            y=dff.loc[dff['Indicator Name'] == yaxis_column_name, 'Value'],
            text=dff.loc[dff['Indicator Name'] == yaxis_column_name, 'Country Name'],
            mode='markers',
            marker={
                'size': 15,
                'opacity': 0.5,
                'line': {'width': 0.5, 'color': 'white'}
            }
        )],
        'layout': go.Layout(
            xaxis={
                'title': xaxis_column_name,
                'type': 'linear' if xaxis_type == 'Linear' else 'log'
            },
            yaxis={
                'title': yaxis_column_name,
                'type': 'linear' if yaxis_type == 'Linear' else 'log'
            },
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    }

@app.callback(
    Output(component_id='exp-tbl', component_property='children'),
    [Input(component_id='in-exp', component_property='value')]
)
def create_exp_df(x):
    df = pd.DataFrame({'type': ['x**2', 'x**3', '2**x', '3**x', 'x**x'],
                       'value': [x ** 2, x ** 3, 2 ** x, 3 ** x, x ** x]})
    return generate_table(df)

@app.callback(
    Output('cities-dropdown', 'options'),
    [Input('countries-dropdown', 'value')]
)
def set_cities_options(selected_country):
    return [{'label': k, 'value': k} for k in all_options[selected_country]]

@app.callback(
    Output('cities-dropdown', 'value'),
    [Input('cities-dropdown', 'options')])
def set_cities_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('display-selected-values', 'children'),
    [Input('countries-dropdown', 'value'),
     Input('cities-dropdown', 'value')])
def set_display_children(selected_country, selected_city):
    return u'{} is a city in {}'.format(
        selected_city, selected_country,
    )


if __name__ == '__main__':

    app.run_server(debug=True)






