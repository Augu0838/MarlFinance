# %%
# Load libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas import read_csv, set_option
#from pandas.plotting import scatter_matrix
#import seaborn as sns
#from sklearn.preprocessing import StandardScaler
#import datetime
#import math
#from numpy.random import choice
#import random

#from keras.layers import Input, Dense, Flatten, Dropout
#from keras.models import Model
#from keras.regularizers import l2

#import random
#from collections import deque
import matplotlib.pylab as plt

# Import environment and agent
from StockEnv import StockEnvironment
from StockAgent import Agent

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# Load data
url = 'https://github.com/Augu0838/MarlFinance/blob/main/sp500_tickers.csv?raw=true'
df = pd.read_csv(url)
single_list = df.iloc[:, 0].tolist()
single_list = single_list[10:30]
dataset = yf.download(single_list, start="2018-01-01", end="2024-01-01")["Close"]

# Save data to csv
dataset.to_csv(r'data/stock_portfolio.csv')


# %%
##############################
# Declair training variables   
##############################

# Define agent and environment (from agent and env files)
N_ASSETS = len(dataset.columns)
agent = Agent(N_ASSETS)
env = StockEnvironment()

# Choose variables
window_size = 180   # Number of days the agent can 'look back'
episode_count = 50  # Number of training episodes 
batch_size = 32     # Batch size...
rebalance_period = 30 # Only change weight every x days
data_length = len(env.data) # Total number of days in the trading data 

# %%
##############################
# Train the model      
##############################
%capture

for e in range(episode_count):
    
    agent.is_eval = False
    data_length = len(env.data)
    
    returns_history = []
    returns_history_equal = []
    
    rewards_history = []
    equal_rewards = []
    
    actions_to_show = []
    
    print("Episode " + str(e) + "/" + str(episode_count), 'epsilon', agent.epsilon)

    rnd_start_day = np.random.randint(window_size+1, data_length-window_size-1) # set random start day with minimum 180 days before and after. 
    s = env.get_state(rnd_start_day, window_size) # Get state for the randomly selected start day
    total_profit = 0 

    for t in range(window_size, data_length, rebalance_period): # starts at day=window size (180), runs untill number of days and increments by rebalance_period (30)
        date1 = t-rebalance_period # starting time for current period (rewards sum from this period)
        #correlation from 90-180 days 
        s_ = env.get_state(t, window_size)
        action = agent.act(s_)
        
        actions_to_show.append(action[0])
        
        weighted_returns, reward = env.get_reward(action[0], date1, t)
        weighted_returns_equal, reward_equal = env.get_reward(
            np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

        rewards_history.append(reward)
        equal_rewards.append(reward_equal)
        returns_history.extend(weighted_returns)
        returns_history_equal.extend(weighted_returns_equal)

        done = True if t == data_length else False
        agent.memory4replay.append((s, s_, action, reward, done))
        
        if len(agent.memory4replay) >= batch_size:
            agent.expReplay(batch_size)
            agent.memory4replay = []
            
        s = s_

    rl_result = np.array(returns_history).cumsum()
    equal_result = np.array(returns_history_equal).cumsum()

    # plt.figure(figsize = (12, 2))
    # plt.plot(rl_result, color = 'black', ls = '-')
    # plt.plot(equal_result, color = 'grey', ls = '--')
    # plt.show()

    # plt.figure(figsize = (12, 2))
    # for a in actions_to_show:    
    #     plt.bar(np.arange(N_ASSETS), a, color = 'grey', alpha = 0.25)
    #     plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')
    # plt.show()
    
# %%
##############################
# Test the model     
##############################
%%capture

agent.is_eval = True

actions_equal, actions_rl = [], []
result_equal, result_rl = [], []

for t in range(window_size, len(env.data), rebalance_period):

    date1 = t-rebalance_period
    s_ = env.get_state(t, window_size)
    action = agent.act(s_)

    weighted_returns, reward = env.get_reward(action[0], date1, t)
    weighted_returns_equal, reward_equal = env.get_reward(
        np.ones(agent.portfolio_size) / agent.portfolio_size, date1, t)

    result_equal.append(weighted_returns_equal.tolist())
    actions_equal.append(np.ones(agent.portfolio_size) / agent.portfolio_size)
    
    result_rl.append(weighted_returns.tolist())
    actions_rl.append(action[0])


# Append results to series (for rl and equal)
result_equal_vis = [item for sublist in result_equal for item in sublist]
result_rl_vis = [item for sublist in result_rl for item in sublist]

# Plot results
plt.figure()
plt.plot(np.array(result_equal_vis).cumsum(), label = 'Benchmark', color = 'grey',ls = '--')
plt.plot(np.array(result_rl_vis).cumsum(), label = 'Deep RL portfolio', color = 'black',ls = '-')
plt.show()

# %%
#######################################
# Show portfolio allocations   
#######################################

#Plotting the data
import matplotlib
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color='red')

N = len(np.array([item for sublist in result_equal for item in sublist]).cumsum()) 

final_act = len(actions_rl)

current_range = np.arange(0, N)
current_ts = np.zeros(N)
current_ts2 = np.zeros(N)

ts_benchmark = np.array([item for sublist in result_equal[:final_act+1] for item in sublist]).cumsum()
ts_target = np.array([item for sublist in result_rl[:final_act+1] for item in sublist]).cumsum()

t = len(ts_benchmark)
current_ts[:t] = ts_benchmark
current_ts2[:t] = ts_target

current_ts[current_ts == 0] = ts_benchmark[-1]
current_ts2[current_ts2 == 0] = ts_target[-1]

plt.figure(figsize = (12, 10))

plt.subplot(2, 1, 1)
plt.bar(np.arange(N_ASSETS), actions_rl[i], color = 'grey')
plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')

plt.subplot(2, 1, 2)
plt.colormaps = current_cmap
plt.plot(current_range[:t], current_ts[:t], color = 'black', label = 'Benchmark')
plt.plot(current_range[:t], current_ts2[:t], color = 'red', label = 'Deep RL portfolio')
plt.plot(current_range[t:], current_ts[t:], ls = '--', lw = .1, color = 'black')
plt.autoscale(False)
plt.ylim([-1, 1])
plt.legend()

# for i in range(0, len(actions_rl)):
#     current_range = np.arange(0, N)
#     current_ts = np.zeros(N)
#     current_ts2 = np.zeros(N)

#     ts_benchmark = np.array([item for sublist in result_equal[:i+1] for item in sublist]).cumsum()
#     ts_target = np.array([item for sublist in result_rl[:i+1] for item in sublist]).cumsum()

#     t = len(ts_benchmark)
#     current_ts[:t] = ts_benchmark
#     current_ts2[:t] = ts_target

#     current_ts[current_ts == 0] = ts_benchmark[-1]
#     current_ts2[current_ts2 == 0] = ts_target[-1]

#     plt.figure(figsize = (12, 10))

#     plt.subplot(2, 1, 1)
#     plt.bar(np.arange(N_ASSETS), actions_rl[i], color = 'grey')
#     plt.xticks(np.arange(N_ASSETS), env.data.columns, rotation='vertical')

#     plt.subplot(2, 1, 2)
#     plt.colormaps = current_cmap
#     plt.plot(current_range[:t], current_ts[:t], color = 'black', label = 'Benchmark')
#     plt.plot(current_range[:t], current_ts2[:t], color = 'red', label = 'Deep RL portfolio')
#     plt.plot(current_range[t:], current_ts[t:], ls = '--', lw = .1, color = 'black')
#     plt.autoscale(False)
#     plt.ylim([-1, 1])
#     plt.legend()

# %%
#######################################
# Compute final sharp ratio      
#######################################

import statsmodels.api as sm
from statsmodels import regression
def sharpe(R):
    r = np.diff(R)
    sr = r.mean()/r.std() * np.sqrt(252)
    return sr

def print_stats(result, benchmark):

    sharpe_ratio = sharpe(np.array(result).cumsum())
    returns = np.mean(np.array(result))
    volatility = np.std(np.array(result))
    
    X = benchmark
    y = result
    x = sm.add_constant(X)
    model = regression.linear_model.OLS(y, x).fit()    
    alpha = model.params[0]
    beta = model.params[1]
    
    return np.round(np.array([returns, volatility, sharpe_ratio, alpha, beta]), 4).tolist()

print('EQUAL', print_stats(result_equal_vis, result_equal_vis))
print('RL AGENT', print_stats(result_rl_vis, result_equal_vis))
# %%
