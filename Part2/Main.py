#%%
########### Import packages ###########
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import torch
import time
import os
import plots as p


from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent

from portfolio_mng_new import external_weights_new
from func import download_close_prices

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
device = torch.device("cpu") 
print(f"Using device: {device}") 

#%% --------------------------------------------------------------------------
# 0.  ──‑‑‑ INPUTS  ‑‑‑——————————————————————————————————————————————————
num_agents = 1
stocks_per_agent = 98*5
num_stocks = num_agents * stocks_per_agent

window_size = 20
episodes = 200

#%% --------------------------------------------------------------------------
# 1.  ──‑‑‑ DATA  ‑‑‑——————————————————————————————————————————————————
start_day = "2022-05-01"
cache_file = f"cached_data_{num_stocks}_{start_day}.pkl"

# Try loading cached data
if os.path.exists(cache_file):
    print("Loading cached data...")
    data = pd.read_pickle(cache_file)
else:
    print("Downloading fresh data...")
    tickers = pd.read_csv(
        "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
    ).iloc[:num_stocks+10, 0].tolist()

    data = download_close_prices(tickers, start_day=start_day, period_days=365*3)
    data.dropna(inplace=True)

    if data.shape[1] < num_stocks:
        raise ValueError(f"Only {data.shape[1]} tickers available, but {num_stocks} required.")
    else:
        data = data.iloc[:, :num_stocks]

    # Save to file
    data.to_pickle(cache_file)
    print("Data cached to:", cache_file)

# 80 / 20 chronological random split
total_rows = len(data)
test_len = int(total_rows * 0.20)
max_start = total_rows - test_len

# Ensure training data is long enough
min_train_rows = window_size + 1
#test_start = random.randint(min_train_rows, max_start)
# change test start to include 2025-05-01
test_start = max_start


# Corrected Slicing
train_data = data.iloc[:test_start]  # ← up to the start of test set
test_data  = data.iloc[test_start - window_size : test_start + test_len]
# Print forst day in test data
print("First day in test data:", test_data.index[0])

print('Training and test data loaded')

#%% --------------------------------------------------------------------------
# 2.  ──‑‑‑ INITIALIZE ENV AND AGENT  ‑‑‑———————————————————————————————————————
# ------------------------------------------------------------------
external_trader = external_weights_new(data, 
                                  momentum_lookback=50,
                                  vol_lookback=20,
                                  meanrev_short=5, 
                                  meanrev_long=20,
                                  top_quantile=0.1,
                                  band=1.0)

env_train = MultiAgentPortfolioEnv(
    train_data, num_agents, window_size, external_trader=external_trader
)
env_test  = MultiAgentPortfolioEnv(
    test_data,  num_agents, window_size, external_trader=external_trader
)

env = env_train         

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
        ag.actor.train(mode=train)
        ag.critic.train(mode=train)

    t0 = time.perf_counter()      

    for ep in range(1, episodes + 1):
        ep_t0 = time.perf_counter()             
        state = env.reset()
        done, step, total_r = False, 0, np.zeros(num_agents)
        ep_actions = [] # reset the stored actions when evaluating

        while not done:
            actions           = [ag.act(st) for ag, st in zip(agents, state)]
            nxt, r, done, _   = env.step(actions)
            total_r          = r + total_r
            step             = 1 + step

            if not train:
                ep_actions.append(np.vstack(actions))  # Stack agent actions for current step

            if train:
                for i, ag in enumerate(agents):
                    ag.rewards.append(r[i])
                    if step % 5 == 0:
                        ag.update_single()

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


    elapsed = time.perf_counter() - t0          # ➎  total duration
    if train:
        print(f"\n'TRAIN finished in {elapsed:,.2f} s ({elapsed/60:.1f} min)")

    return (np.vstack(metrics), elapsed, action_logs) if not train else (np.vstack(metrics), elapsed, sharpe_per_episode)

#%% --------------------------------------------------------------------------
# 4.  ──‑‑‑ TRAIN  ‑‑‑———————————————————————————————————————————————————
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = env_train
train_scores, _, sharpe_per_episode = run(episodes=episodes, train=True)

# Save model
for i, ag in enumerate(agents):
    torch.save({
        'actor_state_dict': ag.actor.state_dict(),
        'critic_state_dict': ag.critic.state_dict()
    }, f"agent_{i}_checkpoint.pth")

print('Model trained')

#%% ----------------------------------------------------------------------
# 5.  ──‑‑‑ EVALUATE  ‑‑‑—————————————————————————————————————————————
eval_periods = 15
mean_sharpe_comb_list = []
mean_sharpe_ext_list = []

def rolling_sharpe(returns, window=window_size):
    returns = pd.Series(returns)
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std() + 1e-6
    return (mean / std).values

for i in range(eval_periods):
    print(f"Evaluation {i+1}/{eval_periods}")
    # Load memory
    for i, ag in enumerate(agents):
        checkpoint = torch.load(f"agent_{i}_checkpoint.pth", map_location=device)
        ag.actor.load_state_dict(checkpoint['actor_state_dict'])
        ag.critic.load_state_dict(checkpoint['critic_state_dict'])

    env = env_test                             
    eval_scores, _, action_logs = run(episodes=1, train=False)
    print("Evaluation Sharpe:", eval_scores[0][0].round(4))

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

    external_daily_returns = []
    for t in range(max_len):
        date = eval_dates[t]
        if date in external_trader.index:
            weights = external_trader.loc[date].values
            r = np.dot(ret_window[t], weights)
            external_daily_returns.append(r)
        else:
            external_daily_returns.append(0.0)  # fallback if date not available

    sharpe_combined = np.nan_to_num(rolling_sharpe(combined_daily_returns), nan=0.0)
    sharpe_external = np.nan_to_num(rolling_sharpe(external_daily_returns), nan=0.0)

    mean_sharpe_combined = np.mean(sharpe_combined)
    mean_sharpe_external = np.mean(sharpe_external)

    mean_sharpe_comb_list.append(mean_sharpe_combined)
    mean_sharpe_ext_list.append(mean_sharpe_external)

#%%
p.sharpe_ratios(mean_sharpe_comb_list, mean_sharpe_ext_list)
print("Combined Sharpe Ratio:", mean_sharpe_combined.round(4))
print("External Sharpe Ratio:", mean_sharpe_external.round(4))

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

sharpe_combined = np.nan_to_num(rolling_sharpe(combined_daily_returns), nan=0.0)
sharpe_external = np.nan_to_num(rolling_sharpe(external_daily_returns), nan=0.0)

mean_sharpe_combined = np.mean(sharpe_combined)
mean_sharpe_external = np.mean(sharpe_external)

print("Combined Sharpe Ratio:", mean_sharpe_combined.round(4))
print("External Sharpe Ratio:", mean_sharpe_external.round(4))

#%% --------------------------------------------------------------------------
# 7.  ──‑‑‑ Plot  --------------------------------------

#p.plot_training_sharpe(sharpe_per_episode)

p.sharpe_ratios(sharpe_combined, sharpe_external)

p.sharp_difference(sharpe_combined, sharpe_external)

p.cumulative_returns(eval_dates, combined_daily_returns, external_daily_returns)

p.histogram(combined_daily_returns, external_daily_returns)

p.weights_plot(action_logs, external_trader, test_data, date)



# %%
