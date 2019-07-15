"""
A/B Testing with Python - Walkthrough Udacity's Cource Final Project
https://www.kaggle.com/tammyrotem/ab-tests-with-python

"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import math as mt


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 1000)


con_df = pd.read_csv(r'E:\shared\AdHoc\EddieDeane\relearning_python\data\control_data.csv')
exp_df = pd.read_csv(r'E:\shared\AdHoc\EddieDeane\relearning_python\data\experiment_data.csv')


con_df
exp_df

def get_z_score(alpha):
    return norm.ppf(alpha)


met_col = 'Pageviews'


for met_col in ('Pageviews', 'Clicks'):
    con_pv = con_df[met_col].sum()
    exp_pv = exp_df[met_col].sum()

    p = .5
    alpha = .05
    p_hat = round(con_pv / (con_pv + exp_pv), 4)

    sd = mt.sqrt(p * (1-p) / (con_pv+exp_pv))

    z_sco = get_z_score(1-(alpha / 2))
    me = round(z_sco * sd, 4)
    print(f'Is metric {met_col} between {p + me} and {p - me}? Metric: {p_hat}' )


round(con_df['Clicks'].sum() / con_df['Pageviews'].sum(), 4)
round(exp_df['Clicks'].sum() / exp_df['Pageviews'].sum(), 4)


con_df.dropna(inplace=True)
exp_df.dropna(inplace=True)

con_df['Date'].min()
con_df['Date'].max()

exp_df['Date'].min()
exp_df['Date'].max()


cont_enro = con_df['Enrollments'].sum()
exp_enro = exp_df['Enrollments'].sum()


cont_cli = con_df['Clicks'].sum()
exp_cli = exp_df['Clicks'].sum()

cont_gc = cont_enro / cont_cli
exp_gc = exp_enro / exp_cli


cont_gc
exp_gc

pool_gc = (cont_enro + exp_enro) / (cont_cli + exp_cli)
pool_gc
pool_gc_sd = mt.sqrt(pool_gc * (1 - pool_gc) * (1/cont_cli + 1/exp_cli))
pool_gc_sd
gc_me = round(get_z_score(1-alpha/2) * pool_gc_sd, 4)
gc_me
gc_dif = round(exp_gc - cont_gc, 4)
gc_dif
gc_dif - gc_me
gc_dif + gc_me

con_df
full_df = pd.merge(con_df, exp_df, how='inner', on='Date', suffixes=('_cont', '_exp'))
full_df



def get_prob(x,n):
    p=round(mt.factorial(n)/(mt.factorial(x)*mt.factorial(n-x))*0.5**x*0.5**(n-x),4)
    return p

def get_2side_pvalue(x,n):
    p=0
    for i in range(0,x+1):
        p=p+get_prob(i,n)
    return 2*p


full_df['gc_cont'] = full_df['Enrollments_cont'] / full_df['Clicks_cont']
full_df['gc_exp'] = full_df['Enrollments_exp'] / full_df['Clicks_exp']

full_df['gc'] = full_df['gc_cont'] < full_df['gc_exp']


full_df['nc_cont'] = full_df['Payments_cont'] / full_df['Clicks_cont']
full_df['nc_exp'] = full_df['Payments_exp'] / full_df['Clicks_exp']

full_df['nc'] = full_df['nc_cont'] < full_df['nc_exp']
full_df

full_df['gc'].sum()
full_df['nc'].sum()


round(get_2side_pvalue(full_df['gc'].sum(), len(full_df)), 4)
round(get_2side_pvalue(full_df['nc'].sum(), len(full_df)), 4)







































