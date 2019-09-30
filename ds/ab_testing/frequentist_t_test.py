"""
frequentist t-test

assume the data is Gaussian

if you assume the data is not Gaussian, you can use Kolmogorov-Smirnov test, Kruskal-Wallis test,
and Mann-Whitney U Test

"""

import numpy as np
from scipy import stats
import pandas as pd

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


n = 100
# sigma * np.random.randn() + mu
a = 1 * np.random.randn(n) + 1.2
b = 1 * np.random.randn(n) + 0

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)
s = np.sqrt((var_a + var_b) / 2)
t = (a.mean() - b.mean()) / (s * np.sqrt(2/n))
df = 2 * n - 2
p = 2 * (1.0 - stats.t.cdf(t, df=df))
t2, p2 = stats.ttest_ind(a, b)
print(f't: {t} \t p: {p}')
print(f't2: {t2} \t p2: {p2}')



def t_stat_with_n(x1, x2, sp, n):
    return (x1 - x2) / (sp * ((2 / n) ** 0.5))



def t_stat_with_two_n(x1, x2, sp, n1, n2):
    return (x1 - x2) / (sp * ((1 / n1 + 1 / n2) ** 0.5))


def welchs_t_stat(x1, x2, s1, s2, n1, n2):
    return (x1 - x2) / ((((s1 ** 2) / n1) + (((s2 ** 2) / n2))) ** 0.5)



t_stat_with_n(10.5, 10, 5, 100)
t_stat_with_two_n(10.5, 10, 5, 100, 100)
welchs_t_stat(10.5, 10, 5, 5, 100, 100)

t_stat_with_n(10.5, 10, 5, 1_000)
t_stat_with_two_n(10.5, 10, 5, 1_000, 1_000)
welchs_t_stat(10.5, 10, 5, 5, 1_000, 1_000)

t_stat_with_n(10.5, 10, 5, 10_000)
t_stat_with_two_n(10.5, 10, 5, 10_000, 10_000)
welchs_t_stat(10.5, 10, 5, 5, 10_000, 10_000)



ad_clicks_df = pd.read_csv('ds/ab_testing/solutions/advertisement_clicks.csv')
ad_clicks_df


ad_clicsk_agg = (ad_clicks_df
    .groupby('advertisement_id', as_index=False)
    .agg({'action': ['count', 'mean', 'std']})
    .flatten_cols(sep='_'))

ad_clicsk_agg.loc[0, 'action_mean']

welchs_t_stat(x1=ad_clicsk_agg.loc[0, 'action_mean'],
              x2=ad_clicsk_agg.loc[1, 'action_mean'],
              s1=ad_clicsk_agg.loc[0, 'action_std'],
              s2=ad_clicsk_agg.loc[1, 'action_std'],
              n1=ad_clicsk_agg.loc[0, 'action_count'],
              n2=ad_clicsk_agg.loc[1, 'action_count'])


a = ad_clicks_df.loc[ad_clicks_df['advertisement_id'] == 'A', 'action'].values
b = ad_clicks_df.loc[ad_clicks_df['advertisement_id'] == 'B', 'action'].values
t2, p2 = stats.ttest_ind(a, b)
t2
p2































