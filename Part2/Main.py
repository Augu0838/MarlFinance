#%%
########### Import packages ###########
import pandas as pd
import numpy as np
from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent
from momentum import MomentumTrader
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pylab as plt
import torch
import time

from func import download_close_prices

#%% --------------------------------------------------------------------------
# 1.  ──‑‑‑ DATA  ‑‑‑——————————————————————————————————————————————————
num_stocks     = 50
tickers = pd.read_csv(
    "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
).iloc[:num_stocks, 0].tolist()

data = download_close_prices(tickers, start_day="2018-01-01", period_days=365*3)
data.dropna(inplace=True)
print(data.head())
print(data.shape)

#%%
########### Initialize environment and agents ###########
num_agents = 5
window_size = 10
stocks_per_agent = num_stocks // num_agents

# Setup external trader
momentum_trader = MomentumTrader(num_stocks=num_stocks)

# Modified env to include trader
env = MultiAgentPortfolioEnv(data, num_agents, window_size, external_trader=momentum_trader)

# Initialize agents (now requires window_size)
agents = [
    PortfolioAgent(stock_count=stocks_per_agent, window_size=window_size)
    for _ in range(num_agents)
]

#%% --------------------------------------------------------------------------
# 3.  ──‑‑‑ CORE LOOP  ‑‑‑———————————————————————————————————————————————
def run(episodes:int, *, train:bool=True):
    """
    Returns
    -------
    metrics : ndarray   shape = (episodes, num_agents)
    elapsed : float     total seconds spent inside this call
    """
    metrics = []
    action_logs = [] if not train else None  # Only collect actions during evaluation
    
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

        ep_elapsed = time.perf_counter() - ep_t0 # ➍  episode duration
        print(f"Episode {ep:>3}: Sharpe → {total_r}  "
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

    return (np.vstack(metrics), elapsed, action_logs) if not train else (np.vstack(metrics), elapsed)

#%% --------------------------------------------------------------------------
# 4.  ──‑‑‑ TRAIN  ‑‑‑———————————————————————————————————————————————————
if __name__ == "__main__":
    train_scores, _ = run(episodes=5, train=True)

    # optional: save checkpoints
    for i, ag in enumerate(agents):
        torch.save(ag.model.state_dict(), f"agent_{i}.pth")

    # ----------------------------------------------------------------------
    # 5.  ──‑‑‑ EVALUATE  ‑‑‑—————————————————————————————————————————————
    # fresh environment for an out‑of‑sample test period if you like
    test_env  = MultiAgentPortfolioEnv(data, num_agents, window_size)
    env = test_env                             # point run() to the test env
    eval_scores, _, action_logs = run(episodes=1, train=False)
    print(f"\nEvaluation Sharpe ratios: {eval_scores[-1]}")

