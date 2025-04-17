# main.py
#%%
########### Import packages ###########
import pandas as pd
import numpy as np
from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent
import yfinance as yf
from datetime import datetime, timedelta

from func import download_close_prices

#%%
########### Get stock data ###########
# Generate mock stock data (500 stocks, 1000 days)
period_days=100
num_stocks=50

tickers = pd.read_csv('/workspaces/MarlFinance/Part2/sp500_tickers.csv')
single_list = tickers.iloc[:, 0].tolist()
single_list = single_list[0:50]
data = download_close_prices(single_list, start_day="2018-01-01", period_days=365*2)
data.dropna(inplace=True)
print(data.head())
print(data.shape)


#%%
########### Initialize environment and agents ###########
num_agents = 5
stocks_pr_agent = int(num_stocks/num_agents)
env = MultiAgentPortfolioEnv(stock_data=data, num_agents=num_agents, window_size=10)
agents = [PortfolioAgent(stock_count=stocks_pr_agent, random_seed=i) for i in range(num_agents)]  # 500 / 5 = 100 per agent

#%%
########### Training loop ###########
episodes = 10
for ep in range(episodes):
    print(f"Episode {ep + 1}")
    states = env.reset()
    done = False
    total_rewards = np.zeros(num_agents)

    while not done:
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        next_states, rewards, env_done, _ = env.step(actions)
        done = env_done
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
        states = next_states

    print(f"Total Sharpe Ratios (shared): {total_rewards}")

# %%
