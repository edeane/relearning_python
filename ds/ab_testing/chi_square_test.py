"""
Chi-Square Test Statistic

chi_square = Sum ( (Observed - Expected) ** 2 ) / Expected

"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1_000)

def flatten_cols(df, sep='_'):
    col_res = []
    for idx, col_names in enumerate(df.columns.values):
        col_names = [i for i in col_names if i != '']
        col_res.append(sep.join(col_names).strip())
    df.columns = col_res
    return df

pd.DataFrame.flatten_cols = flatten_cols


class DataGenerator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
    def next(self):
        click1 = 1 if (np.random.random() < self.p1) else 0
        click2 = 1 if (np.random.random() < self.p1) else 0
        return click1, click2

def get_c2_p_value(T):
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    c2 = det / T[0].sum() * det / T[1].sum() * T.sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - stats.chi2.cdf(x=c2, df=1)
    return c2, p

def run_experiment(p1, p2, N):
    data = DataGenerator(p1, p2)
    p_values = np.empty(N)
    T = np.zeros((2, 2)).astype(np.float32)
    for i in range(N):
        c1, c2 = data.next()
        T[0, c1] += 1
        T[1, c2] += 1
        if 0 in (T[0].sum(), T[1].sum(), T[:, 0].sum(), T[:, 1].sum()):
            p_values[i] = None
        else:
            c2, p = get_c2_p_value(T)
            p_values[i] = p
    plt.plot(p_values)
    plt.plot(np.ones(N) * 0.05)
    plt.show()


run_experiment(0.1, 0.1, 20_000)

ad_df = pd.read_csv('ds/ab_testing/solutions/advertisement_clicks.csv')
ad_agg = ad_df.groupby('advertisement_id', as_index=False).agg({'action': ['sum', 'count']}).flatten_cols()
ad_agg['no_click'] = ad_agg['action_count'] - ad_agg['action_sum']
ad_agg

chi2, p, dof, ex = stats.chi2_contingency(observed=ad_agg[['action_sum', 'no_click']].values,
                                          correction=False)
chi2
p



df = pd.DataFrame({'click': [36, 30],
                   'no_click': [14, 25]},
                  index=['ad_a', 'ad_b'])


get_c2_p_value(df.loc[['ad_a', 'ad_b'], ['click', 'no_click']].values)

df.loc[:, 'total'] = df.sum(axis=1)
df.loc['total', :] = df.sum(axis=0)

df


total_percent_click = df.loc['total', 'click'] / df.loc['total', 'total']
expected_click_ad_a = df.loc['ad_a', 'total'] * total_percent_click
expected_click_ad_b = df.loc['ad_b', 'total'] * total_percent_click

total_percent_no_click = df.loc['total', 'no_click'] / df.loc['total', 'total']
expected_no_click_ad_a = df.loc['ad_a', 'total'] * total_percent_no_click
expected_no_click_ad_b = df.loc['ad_b', 'total'] * total_percent_no_click

df

(((expected_click_ad_a - df.loc['ad_a', 'click']) ** 2) / expected_click_ad_a +
((expected_click_ad_b - df.loc['ad_b', 'click']) ** 2) / expected_click_ad_b +
((expected_no_click_ad_a - df.loc['ad_a', 'no_click']) ** 2) / expected_no_click_ad_a +
((expected_no_click_ad_b - df.loc['ad_b', 'no_click']) ** 2) / expected_no_click_ad_b )


chi2, p, dof, ex = stats.chi2_contingency(observed=df.loc[['ad_a', 'ad_b'], ['click', 'no_click']].values,
                                          correction=False)
chi2
p
oddsratio, p = stats.fisher_exact(df.loc[['ad_a', 'ad_b'], ['click', 'no_click']].values)




