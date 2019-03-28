"""
Learning plotly Dash Callbacks

https://dash.plot.ly/interactive-graphing
"""

import dash

# https://dash.plot.ly/dash-core-components
import dash_core_components as dcc

# https://dash.plot.ly/dash-html-components
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

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

# ---- data -----
df2 = pd.DataFrame({'x': list(range(5)),
                   'y': list(range(5))})

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/cb5392c35661370d95f300086accea51/raw/'
                      '8e0768211f6b747c0db42a9ce9a0937dafcbd8b2/indicators.csv')
df.drop(df.columns[0], inplace=True, axis=1)
available_indicators  = df['Indicator Name'].unique()

df3 = pd.DataFrame({
   f'column {i}': np.random.rand(30) + i*10 for i in range(6)
})


# ---- app ----

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# ---- app layout ----

app.layout = html.Div([
    dcc.Graph(
        id='basic-inter',
        figure={
            'data': [
                {'x': [1,2,3,4],
                 'y': [4,1,3,5],
                 'text': ['a', 'b', 'c', 'd'],
                 'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                 'name': 'Trace 1',
                 'mode': 'markers',
                 'marker': {'size': 12}},
                {'x': [1,2,3,4],
                 'y': [9,4,1,4],
                 'text': ['w', 'x', 'y', 'z'],
                 'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                 'name': 'Trace 2',
                 'mode': 'markers',
                 'marker': {'size': 12}},
            ],
            'layout': {'clickmode': 'event+select'}
        }
    ),

    html.Div(className='row', children=[
        html.Div([
            html.H2('Hover Data'),
            html.P('Mouse over values in the graph.'),
            html.Pre(id='hover-data', style=styles['pre'])
        ], className='three columns'),
        html.Div([
            html.H2('Click Data'),
            html.P('Click on points in the graph.'),
            html.Pre(id='click-data', style=styles['pre'])
        ], className='three columns'),
        html.Div([
            html.H2('Selection Data'),
            html.P('''
            Choose the lasso or rectangle tool in the graph's menu
            bar and then select points in the graph.

            Note that if `layout.clickmode = 'event+select'`, selection data also 
            accumulates (or un-accumulates) selected data if you hold down the shift
            button while clicking.
            '''),
            html.Pre(id='select-data', style=styles['pre'])
        ], className='three columns'),
        html.Div([
            html.H2('Zoom and Relayout Data'),
            html.P('''
            Click and drag on the graph to zoom or click on the zoom
            buttons in the graph's menu bar.
            Clicking on legend items will also fire
            this event.
            '''),
            html.Pre(id='relayout-data', style=styles['pre'])
        ], className='three columns'),
    ]),
    html.Iframe(srcDoc=df2.to_html()),
    html.H1('hello world'),

    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Fertility rate, total (births per woman)'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Life expectancy at birth, total (years)'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            clickData={'points': [{'customdata': 'Japan'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=df['Year'].min(),
        max=df['Year'].max(),
        value=df['Year'].max(),
        marks={str(year): str(year) for year in df['Year'].unique()}
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

    html.Div([
        html.Div(dcc.Graph(id='g1', config={'displayModeBar': False}), className='four columns'),
        html.Div(dcc.Graph(id='g2', config={'displayModeBar': False}), className='four columns'),
        html.Div(dcc.Graph(id='g3', config={'displayModeBar': False}), className='four columns'),
    ], className='row')

])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-inter', 'hoverData')]
)
def disp_hov_data(hoverData):
    return json.dumps(hoverData, indent=2)

@app.callback(
    Output('click-data', 'children'),
    [Input('basic-inter', 'clickData')]
)
def disp_cli_data(x):
    return json.dumps(x, indent=2)

@app.callback(
    Output('select-data', 'children'),
    [Input('basic-inter', 'selectedData')]
)
def disp_sel_data(x):
    return json.dumps(x, indent=2)

@app.callback(
    Output('relayout-data', 'children'),
    [Input('basic-inter', 'relayoutData')]
)
def disp_re_data(x):
    return json.dumps(x, indent=2)


@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('crossfilter-xaxis-type', 'value'),
     Input('crossfilter-yaxis-type', 'value'),
     Input('crossfilter-year--slider', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,
                 year_value):
    dff = df[df['Year'] == year_value]

    return {
        'data': [go.Scatter(
            x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
            text=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
            customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'],
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
            margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
            height=450,
            hovermode='closest'
        )
    }


def create_time_series(dff, axis_type, title):
    return {
        'data': [go.Scatter(
            x=dff['Year'],
            y=dff['Value'],
            mode='lines+markers'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


@app.callback(
    Output('x-time-series', 'figure'),
    [Input('crossfilter-indicator-scatter', 'clickData'),
     Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-xaxis-type', 'value')])
def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
    country_name = hoverData['points'][0]['customdata']
    dff = df[df['Country Name'] == country_name]
    dff = dff[dff['Indicator Name'] == xaxis_column_name]
    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
    return create_time_series(dff, axis_type, title)


@app.callback(
    Output('y-time-series', 'figure'),
    [Input('crossfilter-indicator-scatter', 'clickData'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('crossfilter-yaxis-type', 'value')])
def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
    dff = dff[dff['Indicator Name'] == yaxis_column_name]
    return create_time_series(dff, axis_type, yaxis_column_name)



def highlight(x, y):

    def callback(*selectedDatas):
        selectedpoints = df3.index
        for i, selected_data in enumerate(selectedDatas):
            if selected_data is not None:
                selected_index = [p['customdata'] for p in selected_data['points']]
                if len(selected_index) > 0:
                    selectedpoints = np.intersect1d(selectedpoints, selected_index)


        # set which points are selected with the `selectedpoints` property
        # and style those points with the `selected` and `unselected`
        # attribute. see
        # https://medium.com/@plotlygraphs/notes-from-the-latest-plotly-js-release-b035a5b43e21
        # for an explanation

        figure = {
            'data': [
                {
                    'x': df3[x],
                    'y': df3[y],
                    'text': df3.index,
                    'textposition': 'top',
                    'selectedpoints': selectedpoints,
                    'customdata': df3.index,
                    'type': 'scatter',
                    'mode': 'markers+text',
                    'marker': {
                        'color': 'rgba(0, 116, 217, 0.7)',
                        'size': 12,
                        'line': {
                            'color': 'rgb(0, 116, 217)',
                            'width': 0.5
                        }
                    },
                    'textfont': {
                        'color': 'rgba(30, 30, 30, 1)'
                    },
                    'unselected': {
                        'marker': {
                            'opacity': 0.3,
                        },
                        'textfont': {
                            # make text transparent when not selected
                            'color': 'rgba(0, 0, 0, 0)'
                        }
                    }
                },
            ],
            'layout': {
                'clickmode': 'event+select',
                'margin': {'l': 15, 'r': 0, 'b': 15, 't': 5},
                'dragmode': 'select',
                'hovermode': 'closest',
                'showlegend': False
            }
        }

        # Display a rectangle to highlight the previously selected region
        shape = {
            'type': 'rect',
            'line': {
                'width': 1,
                'dash': 'dot',
                'color': 'darkgrey'
            }
        }
        if selectedDatas[0] and selectedDatas[0]['range']:
            figure['layout']['shapes'] = [dict({
                'x0': selectedDatas[0]['range']['x'][0],
                'x1': selectedDatas[0]['range']['x'][1],
                'y0': selectedDatas[0]['range']['y'][0],
                'y1': selectedDatas[0]['range']['y'][1]
            }, **shape)]
        else:
            figure['layout']['shapes'] = [dict({
                'type': 'rect',
                'x0': np.min(df3[x]),
                'x1': np.max(df3[x]),
                'y0': np.min(df3[y]),
                'y1': np.max(df3[y])
            }, **shape)]

        return figure

    return callback


app.callback(
    Output('g1', 'figure'),
    [Input('g1', 'selectedData'),
     Input('g2', 'selectedData'),
     Input('g3', 'selectedData')]
)(highlight('column 0', 'column 1'))

app.callback(
    Output('g2', 'figure'),
    [Input('g2', 'selectedData'),
     Input('g1', 'selectedData'),
     Input('g3', 'selectedData')]
)(highlight('column 2', 'column 3'))

app.callback(
    Output('g3', 'figure'),
    [Input('g3', 'selectedData'),
     Input('g1', 'selectedData'),
     Input('g2', 'selectedData')]
)(highlight('column 4', 'column 5'))



if __name__ == '__main__':

    app.run_server(debug=True)









