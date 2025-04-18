#%%
########### Import packages ###########
import pandas as pd
import numpy as np
from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pylab as plt

from func import download_close_prices

#%%
########### Get stock data ###########
period_days = 100
num_stocks = 50

tickers = pd.read_csv('https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true')
single_list = tickers.iloc[:, 0].tolist()
single_list = single_list[0:num_stocks]

# Download stock data using your custom function
data = download_close_prices(single_list, start_day="2018-01-01", period_days=365*2)
data.dropna(inplace=True)
print(data.head())
print(data.shape)

#%%
########### Initialize environment and agents ###########
num_agents = 5
window_size = 10
stocks_pr_agent = num_stocks // num_agents

# Initialize environment
env = MultiAgentPortfolioEnv(stock_data=data, num_agents=num_agents, window_size=window_size)

# Initialize agents (now requires window_size)
agents = [
    PortfolioAgent(
        stock_count=stocks_pr_agent,
        window_size=window_size,
        #random_seed=i
    ) for i in range(num_agents)
]

#%%
########### Training loop ###########
episodes = 10
for ep in range(episodes):
    print(f"\nEpisode {ep + 1}")
    states = env.reset()
    done = False
    total_rewards = np.zeros(num_agents)

    step_counter = 0  # Track steps in episode

    while not done:
        # Agents select actions
        actions = [agent.act(state) for agent, state in zip(agents, states)]

        # Environment transitions
        next_states, rewards, env_done, _ = env.step(actions)
        done = env_done

        # Update each agent using the experience
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]

        # Move to next state
        states = next_states
        step_counter += 1

    print(f"Total Sharpe Ratios (shared): {total_rewards}")

    lst_rewards = []
    sum_rewards = sum(total_rewards)/len(total_rewards)
    lst_rewards.append(sum_rewards)

    # Sync target models every 2 episodes
    if ep % 2 == 0:
        for agent in agents:
            agent.sync_target_model()

# %%

print(lst_rewards)

plt.figure()
plt.plot(np.array(lst_rewards).cumsum(), label = 'Deep RL portfolio', color = 'black',ls = '-')
plt.show()
# %%
