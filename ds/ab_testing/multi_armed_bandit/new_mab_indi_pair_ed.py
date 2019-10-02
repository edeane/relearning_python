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

# for each choice bandit type
#   for n_rounds
#     keep track of each round in np array (choice, result) / deque / list
#     aggregate np array to get wins by bandit, trials by bandit

# incorporate
# https://github.com/gSchool/dsi-solns-g49/blob/master/multi-armed-bandit/pair/pair.py


# random choice or use function
# create a pull choice for each bandit type




# bandits



def random_choice(n_bandits):
    '''Random choice'''
    return np.random.randint(n_bandits)

def max_mean(wins, trials):
    ''' Pick the bandit with the current best observed proportion of winning'''
    # make sure to play each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)
    return np.argmax(wins / trials)

def epsilon_greedy(wins, trials, n_bandits, epsilon = 0.1):
    '''Pick a bandit uniformly at random epsilon percent of the time.
    Otherwise pick the bandit with the best observed proportion of winning'''
    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)
    # Explor less than epsilon else pick max
    if random.random() < epsilon:
        return np.random.randint(n_bandits)
    else:
        return np.argmax(wins / trials)

def softmax(wins, trials, n_bandits, tau=0.01):
    ''' Pick an bandit according to the Boltzman Distribution'''
    # Set default value of tau if not provided in init

    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)

    mean = wins / trials
    scaled = np.exp(mean / tau)
    probs = scaled / np.sum(scaled)
    return np.random.choice(range(0, n_bandits), p=probs)

def ucb1(wins, trials, n_round):
    ''' Pick the bandit according to the UCB1 strategy'''
    # Verify that we have attempted each bandit at least once
    if trials.min() == 0:
        return np.argmin(trials)

    means = wins / trials
    confidence_bounds = np.sqrt((2 * np.log(n_round)) / trials)
    upper_confidence_bounds = means + confidence_bounds
    return np.argmax(upper_confidence_bounds)

def bayesian_bandit(wins, trials):
    '''Randomly sample from a beta distribution for each bandit and pick the one with the largest value'''
    samples = [np.random.beta(a=1 + wins_z, b=1 + trials_z - wins_z)
               for wins_z, trials_z in zip(wins, trials)]
    return np.argmax(samples)


opto_choice_dict = ['max_mean', 'random_choice', 'epsilon_greedy', 'softmax', 'ucb1', 'bayesian_bandit']

p_array = np.array([0.03, 0.05, 0.07])
optimal = np.argmax(p_array)

n_bandits = len(p_array)
n_rounds = 100_000

seeds = range(1, 10)
results = {i: [] for i in opto_choice_dict}
results

for seed in seeds:

    np.random.seed(seed)
    random.seed(seed)
    print('\n')
    print(f'seed: {seed}')

    for key in opto_choice_dict:

        # number of wins and trials for each bandit
        wins = np.zeros(n_bandits)
        trials = np.zeros(n_bandits)

        # keep track of each round
        scores = np.zeros(n_rounds)
        choices = np.zeros(n_rounds)

        print(f'running: {key}')

        for n_round in range(n_rounds):

            if key == 'max_mean':
                pull_choice = max_mean(wins, trials)
            elif key == 'random_choice':
                pull_choice = random_choice(n_bandits)
            elif key == 'epsilon_greedy':
                pull_choice = epsilon_greedy(wins, trials, n_bandits)
            elif key == 'softmax':
                pull_choice = softmax(wins, trials, n_bandits)
            elif key == 'ucb1':
                pull_choice = ucb1(wins, trials, n_round)
            elif key == 'bayesian_bandit':
                pull_choice = bayesian_bandit(wins, trials)
            else:
                raise ValueError(f'key not found {key}')

            # keep track of choices made
            choices[n_round] = pull_choice

            # result from pulling
            pull_result = np.random.random() < p_array[pull_choice]

            # update stats for each bandit
            wins[pull_choice] += pull_result
            trials[pull_choice] += 1

            # keep track of each round result
            scores[n_round] = pull_result

        # each bandit pay out
        results[key].append(sum(scores))
        print(f'each bandit payout {np.round(wins / trials, 4)}')
        # our return
        print(f'random payout {(sum(p_array) / 3):.4f}')
        print(f'optimal payout {p_array[optimal]}')
        print(f'our payout {sum(scores) / n_rounds}')
        print(f'total score {sum(scores):n}')




results
for key, val in results.items():
    print(f'{key}: {int(sum(val)):,} {int(np.mean(val)):,}')









