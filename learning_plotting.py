'''
matplotlib
https://matplotlib.org/gallery/index.html
https://matplotlib.org/users/recipes.html

seaborn
https://seaborn.pydata.org/tutorial.html

plotly
https://plot.ly/python/

plotly dash
https://dash.plot.ly/


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')


# seaborn tutorial
# relational
tips = sns.load_dataset('tips')
tips.head()

sns.relplot(x='total_bill', y='tip', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='smoker', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='sex', style='sex', data=tips)
plt.show()


sns.relplot(x='total_bill', y='tip', hue='smoker', style='time', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='size', data=tips)
plt.show()

sns.relplot(x='total_bill', y='tip', hue='size', data=tips, palette=sns.cubehelix_palette())
plt.show()

sns.relplot(x='total_bill', y='tip', hue='size', data=tips, palette=sns.cubehelix_palette(rot=-.5, light=.75))
plt.show()

sns.relplot(x='total_bill', y='tip', size='size', data=tips, sizes=(15, 200))
plt.show()

# line plots

df = pd.DataFrame(dict(time=np.arange(500),
                       value=np.random.randn(500).cumsum()))

df.head()
g = sns.relplot(data=df, x='time', y='value', kind='line')
g
plt.show()



df_rand_x_rand = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=list('xy'))
df_rand_x_rand.head()

sns.relplot(data=df_rand_x_rand, x='x', y='y')
plt.show()

sns.relplot(data=df_rand_x_rand, x='x', y='y', kind='line', sort=False)
plt.show()

# aggregationand representing uncertainty

fmri = sns.load_dataset('fmri')
fmri.head()
fmri.describe()
fmri.info()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line')
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', ci=None)
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', ci='sd')
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', ci='sd')
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', estimator=None)
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', hue='event', estimator=None)
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', kind='line', hue='event')
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', hue='region', style='event', kind='line')
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', hue='region', style='event', kind='line', dashes=False, markers=True)
plt.show()

sns.relplot(data=fmri, x='timepoint', y='signal', hue='event', style='event', kind='line')
plt.show()

sns.relplot(data=fmri[fmri['event'] == 'stim'], x='timepoint', y='signal', hue='region', units='subject',
            estimator=None, kind='line')
plt.show()


# dots
dots = sns.load_dataset('dots')
dots.head()
dots.describe()
dots.groupby('align').agg({'align': 'count'})
dots['align'].value_counts()
dots['choice'].value_counts()
dots.info()
dots.groupby(['align', 'choice']).agg('count')
dots.groupby(['align', 'choice']).count()
dots.pivot_table(index=['align', 'choice'], aggfunc='count')
dots.groupby(['align', 'choice']).agg({'align': 'value_counts', 'choice': 'value_counts'})
dots.groupby(['align', 'choice']).agg({'align': 'count', 'choice': 'count'})
dots.groupby(['align', 'choice']).size()
dots.groupby(['align', 'choice']).agg('size')

dots = dots[dots['align'] == 'dots']

sns.relplot(data=dots, x='time', y='firing_rate', hue='coherence', style='choice', kind='line')
plt.show()

palette = sns.cubehelix_palette(light=.5, n_colors=6)
sns.relplot(data=dots, x='time', y='firing_rate', hue='coherence', style='choice', kind='line', palette=palette,
            size='choice')
plt.show()


df = pd.DataFrame(dict(time=pd.date_range('2017-01-01', periods=500),
                       value=np.random.randn(500).cumsum()))

df.head()
sns.relplot(data=df, x='time', y='value', kind='line')
plt.show()


sns.relplot(data=fmri, x='timepoint', y='signal', hue='subject', col='region', row='event', height=3,
            kind='line', estimator=None)
plt.show()

sns.relplot(data=fmri[fmri['region']=='frontal'], x='timepoint', y='signal', hue='event', col='subject', style='event',
            height=3, col_wrap=5, aspect=.75, linewidth=2.5, kind='line', estimator=None)
plt.show()


# categorical data







# matplotlib bar
# https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

ind = np.arange(len(men_means))
width = 0.35

fig, ax = plt.subplots()

ax.bar()









