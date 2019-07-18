"""
Some A/B Testing Examples

A/B Testing Statistics
https://conversionxl.com/blog/ab-testing-statistics/

"""


# A/B Testing with Python - Walkthrough Udacity's Cource Final Project
# https://www.kaggle.com/tammyrotem/ab-tests-with-python


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







# Functions
# https://medium.com/@henryfeng/handy-functions-for-a-b-testing-in-python-f6fdff892a90


from scipy import stats


def get_power(n, p1, p2, cl):
    alpha = 1 - cl
    qu = stats.norm.ppf(1 - alpha / 2)
    diff = abs(p2 - p1)
    bp = (p1 + p2) / 2

    v1 = p1 * (1 - p1)
    v2 = p2 * (1 - p2)
    bv = bp * (1 - bp)

    power_part_one = stats.norm.cdf((n ** 0.5 * diff - qu * (2 * bv) ** 0.5) / (v1 + v2) ** 0.5)
    power_part_two = 1 - stats.norm.cdf((n ** 0.5 * diff + qu * (2 * bv) ** 0.5) / (v1 + v2) ** 0.5)

    power = power_part_one + power_part_two

    return power



get_power(1000, .1, .12, .95)

def get_sample_size(power, p1, p2, cl, max_n=1000000):
    n = 1 
    while n <= max_n:
        tmp_power = get_power(n, p1, p2, cl)
        if tmp_power >= power: 
            return n
        else:
            n = n + 100
    return "Increase Max N Value"


get_sample_size(.8, .1, .12, .95)

get_power(3901, .1, .12, .95)


def get_pvalue(con_conv, test_conv, con_size, test_size):
    
    lift = -abs(test_conv - con_conv)
    
    scale_one = con_conv * (1 - con_conv) * (1 / con_size)
    scale_two = test_conv * (1 - test_conv) * (1 / test_size)
    scale_val = (scale_one + scale_two) ** 0.5

    p_value = 2 * stats.norm.cdf(lift, loc=0, scale=scale_val)
    return p_value


get_pvalue(.035, .04, 15_000, 15_000)


def get_ci(lift, alpha, sd):
    val = abs(stats.norm.ppf((1 - alpha) / 2))

    lwr_bnd = lift - val * sd
    upr_bnd = lift + val * sd

    return (lwr_bnd, upr_bnd)


get_ci()

-abs(.04 - .035)


test_conv = .04
con_conv = .035
test_size = 15_000
con_size = 15_000
lift_mean = test_conv - con_conv
lift_variance = (1 - test_conv) * test_conv /test_size + (1 - con_conv) * con_conv / con_size
lift_sd = lift_variance ** 0.5
lift_mean
ci = get_ci(lift_mean, 0.95, lift_sd)

f'{round(lift_mean * 100, 4)}%'
[f'{round(i * 100, 2)}%' for i in ci]



# Bayesian A/B Testing
# https://conversionxl.com/blog/ab-testing-statistics/
# https://machinelearningmastery.com/statistical-data-distributions/
# http://pytolearn.csd.auth.gr/d1-hyptest/11/distros.html



# Alpacas vs. Bears
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing/








