#%%
########### Import packages ###########
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import torch
import time

from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent

from portfolio_mng import external_weights
from func import download_close_prices

#%% --------------------------------------------------------------------------
# 0.  ──‑‑‑ INPUTS  ‑‑‑——————————————————————————————————————————————————
num_agents = 5
window_size = 20
episodes = 1000

#%% --------------------------------------------------------------------------
# 1.  ──‑‑‑ DATA  ‑‑‑——————————————————————————————————————————————————
num_stocks     = 200
start_day = "2022-01-01"

tickers = pd.read_csv(
    "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
).iloc[:num_stocks+1, 0].tolist()

data = download_close_prices(tickers, start_day=start_day, period_days=365*5)
data.dropna(inplace=True)

# 80 / 20 chronological random split
total_rows   = len(data)
test_len     = int(total_rows * 0.20)   # 20 %
max_start    = total_rows - test_len

# choose a random start index, but leave a window_size overlap before it
test_start   = random.randint(window_size, max_start)

# slice
train_data = data.iloc[:test_start - window_size]             
test_data  = data.iloc[test_start - window_size : test_start + test_len]

print('Training and test data loaded')

#%% --------------------------------------------------------------------------
# 2.  ──‑‑‑ INITIALIZE ENV AND AGENT  ‑‑‑———————————————————————————————————————
# ------------------------------------------------------------------
stocks_per_agent = num_stocks // num_agents

external_trader = external_weights(num_stocks=num_stocks, start_day=start_day)

env_train = MultiAgentPortfolioEnv(
    train_data, num_agents, window_size, external_trader=external_trader
)
env_test  = MultiAgentPortfolioEnv(
    test_data,  num_agents, window_size, external_trader=external_trader
)

env = env_train          # <‑‑‑ run() always talks to the global “env”

# Initialize agents
agents = [
    PortfolioAgent(stock_count=stocks_per_agent, window_size=window_size)
    for _ in range(num_agents)
]

#%% --------------------------------------------------------------------------
# 3.  ──‑‑‑ TRAINING LOOP FUNCTION  ‑‑‑———————————————————————————————————————
def run(episodes:int, *, train:bool=True):
    """
    Returns
    -------
    metrics : ndarray   shape = (episodes, num_agents)
    elapsed : float     total seconds spent inside this call
    """
    metrics = []
    action_logs = [] if not train else None  # Only collect actions during evaluation
    sharpe_per_episode = [] # Used for generating sharpe over episodes plot
    
    for ag in agents:
        ag.model.train(mode=train)

    t0 = time.perf_counter()      # ➋  start global timer

    for ep in range(1, episodes + 1):
        ep_t0 = time.perf_counter()              # ➌  start episode timer
        state = env.reset()
        done, step, total_r = False, 0, np.zeros(num_agents)
        ep_actions = [] # reset the stored actions when evaluating

        while not done:
            actions           = [ag.act(st) for ag, st in zip(agents, state)]
            nxt, r, done, _   = env.step(actions)
            total_r          += r
            step             += 1

            if not train:
                ep_actions.append(np.vstack(actions))  # Stack agent actions for current step

            if train:
                for i, ag in enumerate(agents):
                    ag.rewards.append(r[i])

            state = nxt

        mean_r = total_r/step + 0.00001
        ep_elapsed = time.perf_counter() - ep_t0 # ➍  episode duration

        if train: 
            sharpe_per_episode.append(mean_r[0])

        print(f"Episode {ep:>3}: Sharpe → {mean_r[0].round(4)}  "
              f"(took {ep_elapsed:5.2f}s)")

        metrics.append(total_r)
        
        if not train:
            action_logs.append(ep_actions)  # Save episode's actions

        if train:
            for ag in agents:
                ag.update()


    elapsed = time.perf_counter() - t0          # ➎  total duration
    print(f"\n{'TRAIN' if train else 'EVAL '} finished "
          f"in {elapsed:,.2f} s "
          f"({elapsed/60:.1f} min)")

    return (np.vstack(metrics), elapsed, action_logs) if not train else (np.vstack(metrics), elapsed, sharpe_per_episode)

#%% --------------------------------------------------------------------------
# 4.  ──‑‑‑ TRAIN  ‑‑‑———————————————————————————————————————————————————
env = env_train
train_scores, _, sharpe_per_episode = run(episodes=episodes, train=True)

# optional: save checkpoints
for i, ag in enumerate(agents):
    torch.save(ag.model.state_dict(), f"agent_{i}.pth")

print('Model trained')

#%% ----------------------------------------------------------------------
# 5.  ──‑‑‑ EVALUATE  ‑‑‑—————————————————————————————————————————————
env = env_test                             
eval_scores, _, action_logs = run(episodes=1, train=False)
print("Evaluation Sharpe:", eval_scores[-1])

#%% --------------------------------------------------------------------------
# 6.  ──‑‑‑‑ PROCESS RESULTS  ‑‑‑—————————————————————————————————————————

# Extract necessary data
returns = np.diff(test_data.values, axis=0) / test_data.values[:-1]  # shape (T-1, S)
dates = test_data.index[1:]  # align with returns

# Determine actual usable length (safe length after start_idx)
eval_len = len(action_logs[0])  # number of timesteps in the episode
start_idx = env.window_size
max_len = min(eval_len, len(dates) - start_idx)

# Align dates and return windows safely
eval_dates = dates[start_idx : start_idx + max_len]
ret_window = returns[start_idx : start_idx + max_len]

# ------------------ 1. Combined portfolio returns ------------------

combined_daily_returns = []
for t in range(max_len):
    step_actions = action_logs[0][t]
    agent_weights = np.vstack(step_actions).flatten()
    date = eval_dates[t]
  
    if date in external_trader.index:
        ext_weights = external_trader.loc[date].values
        combo_weights = agent_weights + ext_weights
        combo_weights /= combo_weights.sum()
    else:
        combo_weights = agent_weights

    r = np.dot(ret_window[t], combo_weights)
    combined_daily_returns.append(r)

# ------------------ 2. External-only portfolio returns ------------------

external_daily_returns = []
for t in range(max_len):
    date = eval_dates[t]
    if date in external_trader.index:
        weights = external_trader.loc[date].values
        r = np.dot(ret_window[t], weights)
        external_daily_returns.append(r)
    else:
        external_daily_returns.append(0.0)  # fallback if date not available

# ------------------ 3. Rolling Sharpe (10-day window) ------------------

def rolling_sharpe(returns, window=window_size):
    returns = pd.Series(returns)
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std() + 1e-6
    return (mean / std).values

sharpe_combined = rolling_sharpe(combined_daily_returns)
sharpe_external = rolling_sharpe(external_daily_returns)

#%% --------------------------------------------------------------------------
# 7.  ──‑‑‑ Plot  --------------------------------------
import plots as p

p.plot_training_sharpe(sharpe_per_episode)

p.sharpe_ratios(sharpe_combined, sharpe_external)

p.sharp_difference(sharpe_combined, sharpe_external)

p.cumulative_returns(eval_dates, combined_daily_returns, external_daily_returns)

p.histogram(combined_daily_returns, external_daily_returns)

# %%
