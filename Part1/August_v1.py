import yfinance as yf

etf_tickers = ["SPY", "ICLN", "XLI"]  # Example ETFs
etf_data = yf.download(etf_tickers, start="2018-01-01", end="2025-01-01")["Close"]
etf_data.dropna(inplace=True)

import pandas as pd

url = 'https://github.com/Augu0838/MarlFinance/blob/main/sp500_tickers.csv?raw=true'
df = pd.read_csv(url)
single_list = df.iloc[:, 0].tolist()
single_list = single_list[0:3]

data = yf.download(single_list, start="2018-01-01", end="2025-01-01")["Close"]
data.dropna(inplace=True)
#data = data.drop(columns = ['High', 'Low', 'Open', 'Volume'])


def sharpe_ratio(returns, risk_free_rate=0.04):
    """
    Compute Sharpe ratio from a 1D array (or Series) of returns.
    """
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(252)



import numpy as np
import pickle
import matplotlib.pyplot as plt
from TradingEnv import StockTradingEnv

def run(episodes, is_training = True):
    
    # Define invironment here called env
    env = StockTradingEnv(data)
    n_days = env.n_steps     # number of date steps
    act_space = env.action_space.n         # 3 * n_stocks

    if is_training:
        q = np.zeros((n_days, act_space))
    else: 
        f = open('qTable.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1
    epsilon_decay_rate = 0.0001

    rng = np.random.default_rng()
    rewards_per_episode = []
    holdings_per_episode = []

    for i in range(episodes):
        state, _ = env.reset()
        day_index = int(state[0])
        terminated = False
        episode_rewards = 0

        while not terminated:
            if is_training and rng.random() < epsilon:
                # action = env.action_space.sample()
                action = rng.integers(act_space)
            else:
                action = np.argmax(q[day_index,:])
            
            new_state, reward, terminated,_ = env.step(action)

            # Extract the new day index
            new_day_index = int(new_state[0])

            # Q-update
            if is_training:
                #q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action])
                q[day_index, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_day_index, :]) - q[day_index, action]
                )

            episode_rewards += reward
            day_index = new_day_index

        # Decay epsilon
        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, 0.01)
        
        holdings_per_episode.append(env.holdings.copy())
        rewards_per_episode.append(episode_rewards)
        
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])

    plt.plot(rewards_per_episode)
    plt.savefig('plot_holdings.png')

    plt.plot(sum_rewards)
    plt.savefig('plot_rewards.png')

    if is_training:
        f = open('qTable.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(episodes=50, is_training = True)
