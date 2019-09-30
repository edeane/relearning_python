"""
Multi Armed Bandit Review from Galvanize
Eddie Deane

python distributions
http://pytolearn.csd.auth.gr/d1-hyptest/11/distros.html

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
from pathlib import Path

full_filepath = Path('ds/ab_testing/multi_armed_bandit')


df_a = pd.read_csv(full_filepath / 'data/siteA.txt', header=None, names=['site_a'])
df_b = pd.read_csv(full_filepath / 'data/siteB.txt', header=None, names=['site_b'])
df_a
df_b

# alpha is the number of clicks
# beta is the number of non-clicks


df_a['site_a'].mean()
df_b['site_b'].mean()

df_a['site_a'].std()
df_b['site_b'].std()

views = [50, 100, 200, 400, 800]
t_stats = []
t_p_vals = []

for view in views:

    # Slice Data for View
    data_a = df_a['site_a'].values[:view]
    data_b = df_b['site_b'].values[:view]
    mean_a = data_a.mean()
    std_a = data_a.std()
    mean_b = data_b.mean()
    std_b = data_b.std()
    
    # T Test
    t_stat, t_p_val = stats.ttest_ind(data_a, data_b, equal_var=False)
    t_stats.append(t_stat)
    t_p_vals.append(t_p_val)

    # Plot T Distribution
    stats_t_a = stats.t(df=view-1, loc=mean_a, scale=std_a)
    t_x_a = np.linspace(stats_t_a.ppf(0.0001), stats_t_a.ppf(0.9999), 1_000)
    t_y_a = stats_t_a.pdf(t_x_a)

    stats_t_b = stats.t(df=view-1, loc=mean_b, scale=std_b)
    t_x_b = np.linspace(stats_t_b.ppf(0.0001), stats_t_b.ppf(0.9999), 1_000)
    t_y_b = stats_t_b.pdf(t_x_b)

    fig, ax = plt.subplots()
    sns.lineplot(t_x_a, t_y_a, label=f'A')
    sns.lineplot(t_x_b, t_y_b, label=f'B')
    plt.title(f'T Dist Views: {view} T Stat: {t_stat:.02f} P Values {t_p_val:.03f}')
    plt.show()

    # Normal
    stats_norm_a = stats.norm(loc=mean_a, scale=std_a)
    norm_x_a = np.linspace(stats_norm_a.ppf(0.0001), stats_norm_a.ppf(0.9999), 1_000)
    norm_y_a = stats_norm_a.pdf(norm_x_a)

    stats_norm_b = stats.norm(loc=mean_b, scale=std_b)
    norm_x_b = np.linspace(stats_norm_b.ppf(0.0001), stats_norm_b.ppf(0.9999), 1_000)
    norm_y_b = stats_norm_b.pdf(norm_x_b)

    fig, ax = plt.subplots()
    sns.lineplot(norm_x_a, norm_y_a, label=f'A')
    sns.lineplot(norm_x_b, norm_y_b, label=f'B')
    plt.title(f'Norm Dist Views: {view} T Stat: {t_stat:.02f} P Values {t_p_val:.03f}')
    plt.show()
    
    # Bayes with Beta
    alpha_a = data_a.sum()
    beta_a = len(data_a) - alpha_a
    stats_beta_a = stats.beta(a=alpha_a, b=beta_a)
    beta_x_a = np.linspace(stats_beta_a.ppf(0.0001), stats_beta_a.ppf(0.9999), 1_000)
    beta_y_a = stats_beta_a.pdf(beta_x_a)

    alpha_b = data_b.sum()
    beta_b = len(data_b) - alpha_b
    stats_beta_b = stats.beta(a=alpha_b, b=beta_b)
    beta_x_b = np.linspace(stats_beta_b.ppf(0.0001), stats_beta_b.ppf(0.9999), 1_000)
    beta_y_b = stats_beta_b.pdf(beta_x_b)

    fig, ax = plt.subplots()
    sns.lineplot(beta_x_a, beta_y_a, label=f'A')
    sns.lineplot(beta_x_b, beta_y_b, label=f'B')
    plt.title(f'Beta Dist Views: {view} A Perc: {alpha_a / view:.2%} B Per: {alpha_b / view:.2%}')
    plt.show()














