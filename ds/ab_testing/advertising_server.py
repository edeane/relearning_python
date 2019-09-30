"""
Advertising server.

"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
from flask import Flask, jsonify, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

plt.interactive(False)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1_000)
np.set_printoptions(suppress=True)

def flatten_cols(df, sep='_'):
    col_res = []
    for idx, col_names in enumerate(df.columns.values):
        col_names = [i for i in col_names if i != '']
        col_res.append(sep.join(col_names).strip())
    df.columns = col_res
    return df

pd.DataFrame.flatten_cols = flatten_cols



app = Flask(__name__)

class Bandit:
    def __init__(self, name):
        self.name = name
        self.clks = 0
        self.views = 0

    def sample(self):
        a = 1 + self.clks
        b = 1 + self.views - self.clks
        return np.random.beta(a, b)

    def add_click(self):
        self.clks += 1

    def add_view(self):
        self.views += 1

        if self.views % 10 == 0:
            print(f'{self.name}: clks={self.clks}, views={self.views} percent: {self.clks / self.views:.2%}')


banditA = Bandit('A')
banditB = Bandit('B')

@app.route('/get_ad')
def get_ad():
    if banditA.sample() > banditB.sample():
        ad = 'A'
        banditA.add_view()
    else:
        ad = 'B'
        banditB.add_view()
    return jsonify({'advertisement_id': ad})

@app.route('/click_ad', methods=['POST'])
def click_ad():
    result = 'OK'
    if request.form['advertisement_id'] == 'A':
        banditA.add_click()
    elif request.form['advertisement_id'] == 'B':
        banditB.add_click()
    else:
        result = 'Invalid Input.'

    return jsonify({'result': result})

def create_figure_band(bandits, trial):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x = np.linspace(0, 1, 200)
    for b in bandits:
        aa = 1 + b.clks
        bb = 1 + b.views - b.clks
        y = stats.beta.pdf(x, aa, bb)
        axis.plot(x, y, label=f'bandit: {b.name}')
    axis.set_title(f'Bandit distributions after {trial} trials')
    axis.legend()
    return fig

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

@app.route('/plot.png')
def plot_png():
    views = banditA.views + banditB.views
    bandits = [banditA, banditB]
    # fig = create_figure()
    fig = create_figure_band(bandits=bandits, trial=views)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(port='4242', threaded=True)

