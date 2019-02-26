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
from scipy import stats
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
# https://seaborn.pydata.org/tutorial/categorical.html

tips = sns.load_dataset('tips')
tips.head()

sns.catplot(data=tips, x='day', y='total_bill')
sns.catplot(data=tips, x='day', y='total_bill', jitter=False)
sns.catplot(data=tips, x='day', y='total_bill', kind='swarm')
sns.catplot(data=tips, x='day', y='total_bill', hue='sex', kind='swarm')
sns.catplot(data=tips, x='size', y='total_bill', kind='swarm')
sns.catplot(data=tips, x='smoker', y='tip', order=['No', 'Yes'], kind='swarm')
sns.catplot(data=tips, x='total_bill', y='day', hue='time', kind='swarm')

# boxplots
sns.catplot(data=tips, x='day', y='total_bill', kind='box')
sns.catplot(data=tips, x='day', y='total_bill', kind='box', hue='smoker')
tips['weekend'] = (tips['day'] == 'Sun') | (tips['day'] == 'Sat')
tips.head()
sns.catplot(data=tips, x='day', y='total_bill', kind='box', hue='weekend')


# boxenplot
diamonds = sns.load_dataset('diamonds')
sns.catplot(data=diamonds.sort_values('color'), x='color', y='price', kind='boxen')

# violin plot
sns.catplot(data=tips, x='total_bill', y='day', hue='time', kind='violin', bw=.15, cut=0, split=True)

sns.catplot(x="day", y="total_bill", hue="sex",
            kind="violin", inner="stick", split=True,
            palette="pastel", data=tips)

plt.ioff()
g = sns.catplot(data=tips, x='day', y='total_bill', kind='violin', inner=None)
sns.catplot(data=tips, x='day', y='total_bill', kind='swarm', ax=g.ax, color='k', size=3)
plt.show()



# barplots

titanic = sns.load_dataset('titanic')
sns.catplot(data=titanic, x='sex', y='survived', hue='class', kind='bar')
plt.show()

sns.catplot(data=titanic, x='deck', kind='count', palette='ch:.25')
plt.show()

sns.catplot(data=titanic, y='deck', hue='class', kind='count', palette='pastel')
plt.show()

# point plots (pick up on the differences of slopes...)
sns.catplot(data=titanic, x='sex', y='survived', kind='point', hue='class')
plt.show()

sns.catplot(data=titanic, x='class', y='survived', kind='point', hue='sex')
plt.show()


f, ax = plt.subplots(figsize=(7,3))
sns.countplot(data=titanic, y='deck', color='c')
plt.show()


g = sns.catplot(x="fare", y="survived", row="class",
                kind="box", orient="h", height=1.5, aspect=4,
                data=titanic[titanic['fare']>0], order=[1, 0])
g.set(xscale="log")
g
plt.show()

# distributions

x = np.random.normal(size=100)
sns.distplot(x)
plt.show()

sns.distplot(x, kde=False, rug=True)
plt.show()

sns.distplot(x, kde=False, rug=True, bins=20)
plt.show()

sns.distplot(x, kde=True, hist=False, rug=True, bins=20)
plt.show()

# how kde works (replace guassian curve centered at each value, then sum each curve, then normalize)
x = np.random.normal(size=30)
bandwidth = 1.06 * x.std() * (x.size ** (-1/5))
support = np.linspace(-4, 4, 200)
kernels = []
for x_i in x:
    kernel = stats.norm(x_i, bandwidth).pdf(support)
    kernels.append(kernel)
    plt.plot(support, kernel, color='b')
sns.rugplot(x, linewidth=3)
plt.show()

from scipy.integrate import trapz
density = np.sum(kernels, axis=0)
density /= trapz(density, support)
plt.plot(support, density)
sns.kdeplot(x, shade=True)
plt.show()

sns.kdeplot(x, bw=.2, label='.2')
sns.kdeplot(x, bw=2, label='2')
plt.show()


# scatterplots

mean, cov = [0, 1], [(1, .5), (.5, 1)]
data=np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=['x', 'y'])
print(df.head())
sns.jointplot(data=df, x='x', y='y')
plt.show()

sns.jointplot(data=df, x='x', y='y', kind='hex')
plt.show()

sns.jointplot(data=df, x='x', y='y', kind='kde')
plt.show()

f, ax = plt.subplots(figsize=(6,6))
sns.kdeplot(data=df['x'], data2=df['y'], ax=ax)
sns.rugplot(df['x'], ax=ax)
sns.rugplot(df['y'], ax=ax, vertical=True)
plt.show()


f, ax = plt.subplots(figsize=(6,6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df['x'], df['y'], cmap=cmap, n_levles=60, shade=True)


iris = sns.load_dataset('iris')
sns.pairplot(iris)

g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);
plt.show(g)



# linear relationships
# https://seaborn.pydata.org/tutorial/regression.html#visualizing-linear-relationships










# matplotlib bar
# https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

men_means, men_std = (20, 35, 30, 35, 27), (2, 3, 4, 1, 2)
women_means, women_std = (25, 32, 34, 20, 25), (3, 5, 2, 3, 3)

ind = np.arange(len(men_means))
width = 0.35

fig, ax = plt.subplots()

ax.bar()









