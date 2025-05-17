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
import func as f

from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent
from portfolio_mng_new import external_weights_new

device = torch.device("cpu") 
print(f"Using device: {device}") 

#%% --------------------------------------------------------------------------
# 0.  ──‑‑‑ INPUTS  ‑‑‑——————————————————————————————————————————————————
num_agents = 1
stocks_per_agent = 20
num_stocks = num_agents * stocks_per_agent

window_size = 20
episodes = 100

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

    data = f.download_close_prices(tickers, start_day=start_day, period_days=365*3)
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
                    if step % 1 == 0:
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
device = torch.device("cpu")
env = env_train
train_scores, _, sharpe_per_episode = run(episodes=episodes, train=True)

# Save model
for i, ag in enumerate(agents):
    torch.save({
        'actor_state_dict': ag.actor.state_dict(),
        'critic_state_dict': ag.critic.state_dict()
    }, f"agent_{i}_checkpoint.pth")

print('Model trained')

# Plot learning curve
p.learning_curve(
    sharpe_per_episode,
    title=f'Learning Curve - Sharpe Ratio over {episodes} episodes',
    xlabel='Episodes',
    ylabel='Sharpe Ratio'
)

#%% ----------------------------------------------------------------------
# 5.  ──‑‑‑ EVALUATE  ‑‑‑—————————————————————————————————————————————
eval_periods = 20

eval_dict = {
    'Sharpe Combined': pd.DataFrame(),
    'Sharpe External': pd.DataFrame(),
    'Combined Daily Returns': pd.DataFrame(),
    'External Daily Returns': pd.DataFrame(),
}

for i in range(eval_periods):
    print(f"Evaluation {i+1}/{eval_periods}")
    
    # Load models
    for j, ag in enumerate(agents):
        checkpoint = torch.load(f"agent_{j}_checkpoint.pth", map_location=device)
        ag.actor.load_state_dict(checkpoint['actor_state_dict'])
        ag.critic.load_state_dict(checkpoint['critic_state_dict'])

    env = env_test
    eval_scores, _, action_logs = run(episodes=1, train=False)

    result_df = f.process_results(
        df=None,
        test_data=test_data,
        action_logs=action_logs,
        external_trader=external_trader,
        window_size=env.window_size
    )

    for key in eval_dict:
        eval_dict[key][f"eval_{i+1}"] = result_df[key]

# Compute mean variables over all evaluations
mean_series = {
    key: df.mean(axis=1) for key, df in eval_dict.items()
}

# Print mean sharpe rations over the evaluations
print("\nCombined Sharpe Ratio:", mean_series["Sharpe Combined"].mean().round(4))
print("External Sharpe Ratio:", mean_series["Sharpe External"].mean().round(4))


#%% ----------------------------------------------------------------------
# 6.  ──‑‑‑ PLOTS  ‑‑‑—————————————————————————————————————————————

mean_comb_sharpe =  eval_dict["Sharpe Combined"].mean(axis=0).round(4).tolist()
mean_ext_sharpe =  eval_dict["Sharpe External"].mean(axis=0).round(4).tolist()

p.sharpe_ratios(
    mean_comb_sharpe,
    mean_ext_sharpe,
    title=f'Average Sharpe Ratio over {eval_periods} evaluations'
)

p.sharpe_ratios(
    mean_series["Sharpe Combined"],
    mean_series["Sharpe External"],
    title=f'Rolling sharpe ratio over {window_size} days',
)

p.sharp_difference(
    mean_series["Sharpe Combined"],
    mean_series["Sharpe External"],
    title = 'Difference - Rolling sharpe ratio over {window_size} days'
)

# Time-average returns across all evals
mean_comb_returns = eval_dict["Combined Daily Returns"].mean(axis=1)
mean_ext_returns = eval_dict["External Daily Returns"].mean(axis=1)
eval_dates = mean_comb_returns.index

p.cumulative_returns(eval_dates, mean_comb_returns, mean_ext_returns)

p.histogram(mean_comb_returns, mean_ext_returns)

date = eval_dict["Sharpe Combined"].index[-1]
p.weights_plot(action_logs, external_trader, test_data, date)

p.market_returns(test_data)

# %%
