"""



"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 1_000)
pd.set_option('display.width', 1_000)

plt.interactive(False)

# total_rounds_games = 10_000
# rounds_to_games_ratio = 100
# n_rounds = (total_rounds_games * rounds_to_games_ratio) ** 0.5
# n_games = (total_rounds_games / rounds_to_games_ratio) ** 0.5
n_rounds = 1_000
n_games = 10
n_doors = 3
total_rounds_games = n_rounds * n_games
print(f'running simulation with n_rounds: {n_rounds:,}\nn_games: {n_games:,}\n'
      f'total_rounds_games: {total_rounds_games:,}\nn_doors: {n_doors:,}')

data_df = pd.DataFrame(columns=['round', 'game', 'car_door', 'select_door', 'door_open',
                                'switch_door', 'correct_without_switch', 'correct_with_switch'],
                       dtype=int)
import time

start_time = time.time()
for n_round in range(n_rounds):
    perc_complete = (n_round + 1) / n_rounds
    print(f'percent complete: {perc_complete:.0%}', end='\r')
    for n_game in range(n_games):
        doors_available = list(range(n_doors))
        car_door = random.choice(doors_available)
        select_door = random.choice(doors_available)
        doors_available_to_open = [i for i in doors_available if i not in (car_door, select_door)]
        door_open = random.choice(doors_available_to_open)
        doors_available.remove(door_open)
        doors_available.remove(select_door)
        switch_door = random.choice(doors_available)
        correct_without_switch = int(car_door == select_door)
        correct_with_switch = int(car_door == switch_door)
        round_df = pd.DataFrame({'round': [n_round],
                                 'game': [n_game],
                                 'car_door': [car_door],
                                 'select_door': [select_door],
                                 'door_open': [door_open],
                                 'switch_door': [switch_door],
                                 'correct_without_switch': [correct_without_switch],
                                 'correct_with_switch': [correct_with_switch]
                                 })
        data_df = data_df.append(round_df, ignore_index=True)

end_time = time.time()
print(f'completed in {end_time - start_time:.1f} seconds')

data_df.info()
df_agg = data_df.groupby(['game']).agg({'correct_without_switch': ['mean'],
                                        'correct_with_switch': ['mean']})

df_agg.columns = ['_'.join(i).strip() for i in df_agg.columns.ravel()]
df_agg.reset_index(inplace=True)
df_agg

df_agg.agg({'correct_without_switch_mean': ['mean', 'std'],
            'correct_with_switch_mean': ['mean', 'std']})

sns.distplot(df_agg['correct_without_switch_mean'], hist=False, rug=False, label='without_switch')
sns.distplot(df_agg['correct_with_switch_mean'], hist=False, rug=False, label='with_switch')
plt.show()




