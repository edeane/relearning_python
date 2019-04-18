"""
pandas

pandas tutorials
http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html

good pandas cheat sheet
https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

minimally sufficient pandas
https://medium.com/dunder-data/minimally-sufficient-pandas-a8e67f2a2428
Attributes (columns, dtypes, index, shape, T, values)

Aggregation Methods (all, any, count, describe, idxmax, idxmin, max, mean, median, min, mode, nunique, sum, std, var)

Non-Aggretaion Statistical Methods (abs, clip, corr, cov, cummax, cummin, cumprod, cumsum, diff, nlargest, nsmallest,
pct_change, prod, quantile, rank, round)

Subset Selection (head, iloc, loc, tail)

Missing Value Handling (dropna, fillna, interpolate, isna, notna)

Grouping (expanding, groupby, pivot_table, resample, rolling)

Joining Data (append, merge)

Other (asfreq, astype, copy, drop, drop_duplicates, equals, isin, melt, plot, rename, replace, reset_index, sample,
select_dtypes, shift, sort_index, sort_values, to_csv, to_json, to_sql)

Functions (pd.concat, pd.crosstab, pd.cut, pd.qcut, pd.read_csv, pd.read_json, pd.read_sql, pd.to_datetime,
pd.to_timedelta)

10 pandas tricks
https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import timeit
import time

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)

df = pd.DataFrame({'AAA': [4, 5, 6, 7],
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

college_idx.iloc[0,]
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
college[('stabbr', 'satmtmid', 'satvrmid', 'ugds')].head()  # breaks
college[{'stabbr', 'satmtmid', 'satvrmid', 'ugds'}].head()  # works

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
print(emp.head())

emp.columns = [x.lower() for x in emp.columns]
print(emp.head())

emp.groupby(['department', 'gender']).agg({'base_salary': 'mean'}).round(-3).head()
emp.pivot_table(index=['department', 'gender'], values='base_salary', aggfunc='mean').round(-3).head()
emp.pivot_table(index='department', columns='gender', values='base_salary', aggfunc='mean').round(-3).head()

emp.head()
emp.groupby('department').agg({'base_salary': 'mean'}).round(0)
emp.pivot_table(index='department', values='base_salary', aggfunc='mean').round(0)

emp.columns

dept_race_agg = emp.groupby(['department', 'race']).agg({'base_salary': 'mean'}).round(0)
dept_race_agg.reset_index(inplace=True)
dept_race_piv = dept_race_agg.pivot_table(index='department', columns='race', values='base_salary')
dept_race_piv.columns
dept_race_piv.reset_index(inplace=True)
dept_race_piv.head()
dept_race_piv.index
dept_race_agg.head()

# use melt over stack

emp_races = list(dept_race_piv.columns)
emp_races.remove('department')
emp_races

dept_race_piv.head()
dept_race_piv.columns

dept_race_melt = dept_race_piv.melt(id_vars='department', value_vars=emp_races, value_name='avg_salary')
dept_race_melt.head()

# 10 minutes to pandas

# left off here
# http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130101'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(['test', 'train', 'test', 'train']),
                    'F': 'foo'})

df2
df2.dtypes
df2.head()

df = pd.DataFrame(np.random.randn(6, 4),
                  index=pd.date_range('20130101', periods=6),
                  columns=list('ABCD'))

df.columns
df.index
df.to_numpy()
df2.to_numpy()

df.describe()
df.T
df

# sort rows
df.sort_index(axis=0, ascending=False)
df.sort_index(axis=0, ascending=True)

# sort columns
df.sort_index(axis=1, ascending=True)
df.sort_index(axis=1, ascending=False)

df.sort_values(by='B', ascending=True, axis=0)
df.sort_values(by='B', ascending=False, axis=0)

# sum
df.sum(axis=0)  # sum columns
df.sum(axis=1)  # sum rows

df.loc[:, 'B']
df.iloc[:3, ].loc[:, 'C':]

df[df['A'] > 1].loc[:, 'C':]

df.loc[df.loc[:, 'A'] > 1, :]
df[df['A'] > 1].loc[:, 'C':]

df2
df2.isin([0, 3])
df2['true'] = True
df2['false'] = False

# loc by name
# iloc by position
# reccomend the optimized .loc and .iloc for production

type(df['A'])
type(df.loc[:, 'A'])

df.loc['20130101', :]
df.loc[:, ['A', 'B']]
df.loc[:, ['A', 'B']]
df.reset_index(inplace=True)
df.loc[2:5, ['index', 'B', 'D']]

df.iloc[2:4, 0:3]
df.iloc[1:3, :]
df.iloc[:, 1:3]
type(df.iloc[1, 1])
df.iloc[1, 1]
df.loc[1, 'A']

df.set_index(keys='index', inplace=True)
df[df > 0]

df['F'] = range(1, 7)
df

s1 = pd.Series(np.arange(1, 7, 1), index=pd.date_range('20130102', periods=6))
s1
df['G'] = s1
df.loc['20130101', 'G'] = 0
df

df.loc[:, 'H'] = np.array([5] * len(df))
df.shape
len(df)

# missing data

df.loc[:, 'I'] = [None, 1, 2, None, 4, 5]

new_col = [None] * 3 + list(np.random.randint(1, 9, 3))
np.random.shuffle(new_col)
new_col

df.loc[:, 'J'] = new_col
df

df.dropna(how='any')
df.fillna(value=777)
df.isna().any(axis=0)  # down columns
df.isna().any(axis=1)  # acros rows

df.mean(axis=0)
df.mean(axis=1)

np.concatenate([np.random.randint(1, 9, 4), np.array([1, 2, 3])])
np.concatenate([np.random.randint(1, 9, 4), np.array([1, np.nan, 3])])
np.concatenate([np.random.randint(1, 9, 4), np.array([np.nan, np.nan, np.nan])])
a_new_col = np.concatenate([np.random.randint(1, 9, 4), np.array([np.nan] * 3)])
np.random.shuffle(a_new_col)
a_new_col

# apply functions

df.apply(np.cumsum, axis=0)  # down columns
df.apply(np.cumsum, axis=1)  # across rows

s2 = pd.Series(np.random.randint(1, 9, 20))
s2.value_counts()

df = pd.DataFrame({'s2': s2})

df.groupby('s2').agg({'s2': 'count'})
df.loc[:, 's2'].value_counts()
df

df2
df2.agg(['mean', 'min', 'max'])

df = pd.DataFrame(np.random.randn(6, 4),
                  index=pd.date_range('20130101', periods=6),
                  columns=list('ABCD'))

df.loc[:, 'E'] = range(len(df))
df.loc[:, 'F'] = [1] * 3 + [2] * 3
df.agg(['mean', 'min', 'max', 'count'])
# put agg functions in list form to create column names
df_agg = df.groupby('F').agg({'A': ['mean'],
                              'B': ['max'],
                              'C': ['min']})

df_agg
df_agg.columns
['_'.join(x) for x in df_agg.columns.values]

df

df.apply(lambda x: x.max() - x.min(), axis=0)
df.apply(lambda x: x.max() - x.min(), axis=1)


# concat dataframes


df = pd.DataFrame(np.random.randint(1, 10, 40).reshape(10, 4), columns=list('ABCD'))
df_0 = df.iloc[:3]
df_1 = df.iloc[3:7]
df_2 = df.iloc[7:]


pd.concat([df_0, df_1, df_2])


# joins / merges

left = pd.DataFrame({'key': ['foo', 'bar'],
                     'lval': [1, 2]})

right = pd.DataFrame({'key': ['foo', 'bar'],
                     'rval': [4, 5]})

left
right
pd.merge(left, right, on='key')

# append, ignore index

df = pd.DataFrame(np.random.randint(1, 10, 32).reshape(8, 4), columns=list('ABCD'))
df
s = df.iloc[:3, :]
s
df.append(s, ignore_index=True)
pd.concat([df, s], axis=0)

df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar',
                   'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three',
                   'two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})


df.groupby('A').agg('sum')
df.groupby('B').agg('sum')



# categoricals

df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                  "raw_grade": ['a', 'b', 'b', 'a', 'a', 'e']})


df.loc[:, 'grade'] = df.loc[:, 'raw_grade'].astype('category')
df.dtypes
df['grade'].cat.categories = ['vg', 'g', 'vb']
df['grade']




# quick bulit in plotting

ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('2000/01/01', periods=1000))


ts_cumsum = ts.cumsum()
ts_cumsum.plot()
plt.show()

ts.head()
ts_cumsum.head()

list('abcd'.upper())

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list('ABCD'))

df_cumsum = df.apply(np.cumsum, axis=0)


plt.figure()
df_cumsum.plot()
plt.legend(loc='best')
plt.show()


# memory usage

df.info()
df.memory_usage()
df.memory_usage(index=False)



# pandas pydata cheat sheet
# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf

df
df.shape[0]
df.loc[:, 'gender'] = ['male'] * 500 + ['female'] * 500
df.reset_index(inplace=True)
df.columns[0] = 'date'
df.rename(columns={'index': 'date'}, inplace=True)
df.drop('gen_val', inplace=True, axis=1)
df = pd.merge(df, pd.DataFrame({'gender': ['male', 'female'], 'gen_val': [1, 1]}), how='left', on='gender')
df.head()
df = df.pivot_table(index=['date', 'A', 'B', 'C', 'D'], columns='gender', values='gen_val', fill_value=0)
df.reset_index(inplace=True)
df.index
df.columns
df.columns.name
df.head()

df.melt(id_vars=['date', 'A', 'B', 'C', 'D'], value_vars=['female', 'male'])


df = pd.DataFrame({'A': ['a', 'b', 'c'],
                   'B': [1, 3, 5],
                   'C': [2, 4, 6]})
df
df.melt(id_vars=['A'], value_vars=['B', 'C'])


df = pd.DataFrame({'camp': range(1, 6),
                   '2010': range(6, 11),
                   '2011': range(6, 11),
                   '2012': range(6, 11),
                   '2013': range(6, 11)})

df
df_melt = df.melt(id_vars=['camp'], value_vars=['2010', '2011', '2012', '2013'], var_name='year', value_name='sales')
df_melt
df_piv = df_melt.pivot_table(index='camp', columns='year', values='sales').reset_index()
df_piv.melt(id_vars='camp', value_vars=['2010', '2011', '2012', '2013'], value_name='sales')



# concat

df1 = pd.DataFrame({'letter': ['a', 'b'],
                    'number': [1, 2]})

df2 = pd.DataFrame({'letter': ['c', 'd'],
                    'number': [3, 4]})

df3 = pd.DataFrame({'animal': ['bird', 'monkey'],
                    'name': ['polly', 'george']})

pd.concat([df1, df2], axis=0, ignore_index=True)
df_cbind = pd.concat([df1, df3], axis=1)
df_cbind.rename(columns={'animal': 'ani_type', 'name': 'ani_name'}, inplace=True)
df_cbind


# subsetting

df = pd.DataFrame({'a': range(4, 10),
                   'b': range(10, 16),
                   'c': range(16, 22)})

df.loc[df['a'] > 5, ]
df['d'] = [1] * 2 + [2] * 4
df.groupby('d').agg({'d': 'count'})
df['d'].value_counts()
df['area'] = df['a'] * df['b']
pd.qcut(df['a'], 3)



# data analysis with pandas cheat sheet
# http://www.datasciencefree.com/pandas.pdf




# 10 pandas tricks
# https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba


import seaborn as sns

df = sns.load_dataset('mpg')
df.head()
df.info()
df.describe()
df.dtypes

# select dtypes
df.dtypes.value_counts()
df.select_dtypes(include=['float64', 'int64'])
df.select_dtypes(include=['object'])

# copy
df1 = df.copy(deep=True)
df1


# map
df['cylinders'].value_counts(normalize=True)
df['cylinders'].value_counts()


cyl_map = {4: 'four', 8: 'eight', 6: 'six', 3: 'three', 5: 'five'}
df['cylinders'].map(cyl_map)


df.isnull().sum(axis=0)










