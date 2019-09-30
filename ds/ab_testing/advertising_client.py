"""



"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import random
from scipy import stats
from flask import Flask, jsonify, request
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


# get data
df = pd.read_csv('E:/shared/AdHoc/EddieDeane/relearning_python/ds/ab_testing/solutions/advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action'].values
b = b['action'].values

print("a.mean:", a.mean())
print("b.mean:", b.mean())

i = 0
j = 0
count = 0
while i < len(a) and j < len(b):
    # quit when there's no data left for either ad
    r = requests.get('http://localhost:4242/get_ad')
    print(r.content)
    r = r.json()
    if r['advertisement_id'] == 'A':
        action = a[i]
        i += 1
    else:
        action = b[j]
        j += 1

    if action == 1:
        # only click the ad if our dataset determines that we should
        requests.post(
            'http://localhost:4242/click_ad',
            data={'advertisement_id': r['advertisement_id']}
        )

    # log some stats
    count += 1
    if count % 50 == 0:
        print(f"Seen {count} ads, A: {i}, B: {j}")
