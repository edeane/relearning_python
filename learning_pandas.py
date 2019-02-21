"""
pandas
https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428
http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html

# pandas cookbook data
https://github.com/PacktPublishing/Pandas-Cookbook/tree/master/data


"""

import numpy as np
import pandas as pd
from pprint import pprint
import timeit
import time

df = pd.DataFrame({'AAA': [4,5,6,7],
                   'BBB': [10, 20, 30, 40],
                   'CCC': [100, 50, -30, -50]})

print(df)



# minimally Sufficient Pandas


df = pd.read_csv('data/sample_data.csv', index_col=0)

df['state']

col_select = 'favorite food'
df[col_select]
df['favorite food']

# loc by label
# iloc by index



# for performance gains use numpy instead of pandas

college = pd.read_csv('data/college.csv')
college.head()
college.columns = [col.lower() for col in college.columns]


college[college.isna().any(axis=1)]

# only us isna and notna
df.isna()
df.notna()
df.notna().all(axis=1)
df.notna().all(axis=0)


df.isna().all(axis=1)
df.isna().any(axis=1)

college['ugds'] + 100

# returns new df with axis set
college_idx = college.set_index('instnm')
college_idx.head()
# just select columns wanted and drop nas
sats = college_idx[['satmtmid', 'satvrmid']].dropna()
sats.head()


# adding
ugds = college_idx['ugds']
ugds.head()
ugds_operator = ugds + 100
ugds_method = ugds.add(100)

ugds_operator.head() == ugds_method.head()
ugds_operator.equals(ugds_method)


# z score
sat_mean = sats.mean()
sat_mean
sat_std = sats.std()
sat_std


z_op = (sats - sat_mean) / sat_std
z_met = sats.sub(sat_mean).div(sat_std)
z_op.equals(z_met)

college.columns

college_race = college_idx.loc[:, 'ugds_white':'ugds_unkn']
college_race.head()

df_attempt = college_race * ugds
df_attempt.shape

ugds.head()
df_correct = college_race.mul(ugds, axis='index').round(0)
df_correct.head()


"""
+ - add
- - sub and subtract
* - mul and multiply
/ - div, divide and truediv
** - pow
// - floordiv
% - mod

> - gt
< - lt
>= - ge
<= - le
== - eq
!= - ne
"""


college_idx.iloc[0, ]
college_idx.iloc[0:1, ]
college_idx.iloc[:1, ]
college_idx.iloc[:1, :4]
college_idx.loc[college_idx.index[:5], ['city', 'stabbr']]



college_idx[['city', 'stabbr', 'hbcu']].head()
college_idx.loc[:, ['city', 'stabbr', 'hbcu']].head()

college.loc[:5, ['instnm', 'city', 'stabbr', 'hbcu']]
college[['instnm', 'city', 'stabbr', 'hbcu']]


# use built in methods
ugds = college['ugds'].dropna()
ugds.head()
ugds.sum()
sum(ugds)
ugds.max()
max(ugds)


# groupby

college[['stabbr', 'satmtmid', 'satvrmid', 'ugds']].head()
college[('stabbr', 'satmtmid', 'satvrmid', 'ugds')].head() # breaks
college[{'stabbr', 'satmtmid', 'satvrmid', 'ugds'}].head() # works

# 1
college.groupby('stabbr').agg({'satmtmid': 'max'}).head()

college.groupby('stabbr').agg({'satmtmid': ['min', 'max'],
                               'satvrmid': ['min', 'max'],
                               'ugds': 'mean'}).head()



agg_dict = {'satmtmid': ['min', 'max'],
            'satvrmid': ['min', 'max'],
            'ugds': 'mean'}

df = college.groupby(['stabbr', 'relaffil']).agg(agg_dict)
df.columns = ['_'.join(col).strip() for col in df.columns.values]
df.reset_index(inplace=True)
df.head()

# left off here "The similarity between groupby, pivot_table, and crosstab"
# https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428

emp = pd.read_csv('data/employee.csv')
emp.head()













