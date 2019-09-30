"""
Following the Galva Multi Armed Bandit Individual and Pair Assignments

"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import random
from scipy import stats

plt.interactive(False)
plt.style.use('ggplot')

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


path_to_data = Path('ds/ab_testing/multi_armed_bandit/data')


df_a = pd.read_csv(path_to_data / 'siteA.txt', header=None, names=['site_a'])
df_b = pd.read_csv(path_to_data / 'siteB.txt', header=None, names=['site_b'])

#

x = np.linspace(0, 1, 100)
y_uniform = stats.uniform.pdf(x)

def plot_with_fill(x, y, label):
    lines = plt.plot(x, y, label=label, lw=2)
    plt.fill_between(x, 0, y, alpha=0.2, color=lines[0].get_c())
    plt.legend(loc='best')

plot_with_fill(x, y_uniform, 'uni')
plt.show()


# Beta distribution


x = np.linspace(0, 1, 100)
y_uniform = stats.beta(a=1, b=1).pdf(x)

plot_with_fill(x, y_uniform, 'beta')
plt.show()


# alpha = n clicks
# beta = n no-clicks

df_a_agg = df_a.loc[0:49, ].groupby('site_a').agg({'site_a': 'count'})
alpha = df_a_agg.loc[1, 'site_a']
beta = df_a_agg.loc[0, 'site_a']

alpha
beta


x
y_beta = stats.beta(a=alpha, b=beta).pdf(x)

plot_with_fill(x, y_beta, 'first 50')
plot_with_fill(x, y_uniform, 'uniform')
plt.show()



views = []
view_n = 50
for i in range(5):
    views.append(view_n)
    view_n *= 2

views

x = np.linspace(0, 1, 1000)
y = stats.beta(a=1, b=1).pdf(x)
plot_with_fill(x, y, label='prior')
for view in views:
    df_a_agg = df_a.loc[0:view-1, ].groupby('site_a').agg({'site_a': 'count'})
    alpha = df_a_agg.loc[1, 'site_a']
    beta = df_a_agg.loc[0, 'site_a']
    y = stats.beta(a=alpha, b=beta).pdf(x)
    plot_with_fill(x, y, label=f'view {view}')

plt.show()





df_a_agg = df_a.groupby('site_a').agg({'site_a': 'count'})
alpha_a = df_a_agg.loc[1, 'site_a']
beta_a = df_a_agg.loc[0, 'site_a']
y_a = stats.beta(a=alpha_a, b=beta_a).pdf(x)

df_b_agg = df_b.groupby('site_b').agg({'site_b': 'count'})
alpha_b = df_b_agg.loc[1, 'site_b']
beta_b = df_b_agg.loc[0, 'site_b']
y_b = stats.beta(a=alpha_b, b=beta_b).pdf(x)

plot_with_fill(x, y_a, label=f'views a')
plot_with_fill(x, y_b, label=f'views b')
plt.show()


beta_a = stats.beta(a=alpha_a, b=beta_a)
beta_b = stats.beta(a=alpha_b, b=beta_b)

sample_a = beta_a.rvs(size=100_000)
sample_b = beta_b.rvs(size=100_000)
sample_a

prob_better = (sample_a < sample_b).mean()
f'there is a {prob_better:.1%} probability that site b is better than site a'

beta_a.ppf(0.025)
beta_a.ppf(0.975)
f'a {beta_a.ppf(0.025):.4} to {beta_a.ppf(0.975):.4}'
f'b {beta_b.ppf(0.025):.4} to {beta_b.ppf(0.975):.4}'


prob_better = ((sample_a + 0.02) < sample_b).mean()
f'there is a {prob_better:.1%} probability that site b is 2% better than site a'

site_dif = sample_b - sample_a
plt.hist(site_dif, bins=20)
sns.distplot(site_dif, bins=20)
plt.show()


# frequentist ttest

t, p = stats.ttest_ind(df_a['site_a'].values, df_b['site_b'].values)
print(f'p value is {p/2:.2}')

t, p = stats.ttest_ind(df_a['site_a'].values + 0.02, df_b['site_b'].values)
print(f'p value is {p/2:.2}')

# ----- Pair -----

# change to array that concats to bottom with each round


# bandits
seed = 42
np.random.seed(seed)
random.seed(seed)

p_array = np.array([0.02, 0.04, 0.06])
optimal = np.argmax(p_array)


n_bandits = len(p_array)
n_rounds = 1_000
choices_lst = []
score_lst = []


a_arr = np.array([1, 2, 3, 4, 5, 6])
b_arr = np.array([7, 8, 9, 10, 11, 12])

np.r_[a_arr, b_arr]
np.concatenate([a_arr, b_arr])

# number of wins and trials for each bandit
wins = np.zeros(n_bandits)
trials = np.zeros(n_bandits)

# keep track of each round
scores = np.zeros(n_rounds)
choices = np.zeros(n_rounds)


# for each choice bandit type
#   for n_rounds
#     keep track of each round in np array / df

# random choice or use function
# create a pull choice for each bandit type
# choice_dict = {'max_mean': max_mean,
#                'random_choice': random_choice,
#                'epsilon_greedy': epsilon_greedy,
#                'softmax': softmax,
#                'ucb1': ucb1,
#                'bayesian_bandit': bayesian_bandit}

for k in range(n_rounds):

    pull_choice = random.choice([0,1,2])

    # result from pulling
    pull_result = np.random.random() < p_array[pull_choice]

    # update stats
    wins[pull_choice] += pull_result
    trials[pull_choice] += 1

    scores[k] = pull_result
    choices[k] = pull_choice

wins
trials
scores
choices

def max_mean():
    ''' Pick the bandit with the current best observed proportion of winning
    '''
    # make sure to play each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)
    return np.argmax(wins / trials)

wins / trials

for i in range(100):
    print(np.random.randint(n_bandits))

def random_choice():
    return np.random.randint(n_bandits)

def epsilon_greedy():
    '''Pick a bandit uniformly at random epsilon percent of the time.
    Otherwise pick the bandit with the best observed proportion of winning'''
    # Set default value of epsilon if not provided in init
    epsilon = 0.1

    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)
    if random.random() < epsilon:
        # Exploration
        return np.random.randint(n_bandits)
    else:
        return np.argmax(wins / trials)

def softmax():
    ''' Pick an bandit according to the Boltzman Distribution'''
    # Set default value of tau if not provided in init
    tau = 0.01

    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)

    mean = wins / trials
    scaled = np.exp(mean / tau)
    probs = scaled / np.sum(scaled)
    return np.random.choice(range(0, n_bandits), p=probs)


def ucb1(self):
    ''' Pick the bandit according to the UCB1 strategy'''
    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)

    means = wins / trials
    confidence_bounds = np.sqrt((2 * np.log(n_rounds)) / trials)
    upper_confidence_bounds = means + confidence_bounds
    return np.argmax(upper_confidence_bounds)

def bayesian_bandit(self):
    '''Randomly sample from a beta distribution for each bandit and pick the one with the largest value'''
    alpha = 1 + wins
    beta = 1 + trials - wins
    samples = [np.random.beta(a=alpha, b=beta)
               for wins, trials in zip(self.wins, self.trials)]
    return np.argmax(samples)















