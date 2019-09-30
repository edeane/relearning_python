"""



"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy import stats
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










