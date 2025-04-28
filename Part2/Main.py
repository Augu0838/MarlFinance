#%%
########### Import packages ###########
import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import torch
import time

from portfolio_mng import external_weights
from func import download_close_prices

from Env import MultiAgentPortfolioEnv
from shared_ac import SharedCritic, Actor, AC_Agent

#%% --------------------------------------------------------------------------
# 0.  ──‑‑‑ INPUTS  ‑‑‑——————————————————————————————————————————————————
num_agents = 5
window_size = 10
episodes = 10

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

 # ------------------------- initialise agents -------------------------
 
obs_rows = window_size + 1                     # we added 1 row
state_dim = obs_rows * num_stocks

# ------------ build shared critic ------------
critic = SharedCritic(state_dim)
critic_optim = torch.optim.Adam(critic.parameters(), lr=3e-4)

# ------------ build actors -------------------
agents = [
    AC_Agent(obs_dim=obs_rows * stocks_per_agent,
             act_dim=stocks_per_agent,
             actor=Actor(obs_rows * stocks_per_agent, stocks_per_agent))
    for _ in range(num_agents)
 ]

#%% --------------------------------------------------------------------------
# 3.  ──‑‑‑ TRAINING LOOP FUNCTION  ‑‑‑———————————————————————————————————————
def run(episodes, *, train=True,
        gamma=0.99, ent_w=0.01, val_w=0.5):

    episode_rewards, sharpe_log = [], []
    action_logs = [] if not train else None

    for ep in range(episodes):
        state = env.reset()

        # per‑episode buffers
        joint_states, global_rewards = [], []

        done = False
        while not done:
            # -------- actors ----------
            actions = [ag.act(obs) for ag, obs in zip(agents, state)]

            # -------- env -------------
            nxt_state, reward_vec, done, _ = env.step(actions)
            # env returns *identical* reward for every agent → pick first
            r = reward_vec[0]

            # -------- buffers ----------
            joint = np.concatenate([s.flatten() for s in state])
            joint_states.append(joint)
            global_rewards.append(r)

            state = nxt_state

            if not train:
                action_logs.append(actions)   # keep one entry per step
            
        # ---------- critic forward ----------
        state_batch = torch.tensor(
            np.asarray(joint_states, dtype=np.float32))      # ◄ fast, no warning
        values = critic(state_batch)          # requires_grad = True

        # ---------- returns & advantages -----------
        R, returns = 0.0, []
        for r in reversed(global_rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        # stop‑grad for actors, keep graph for critic
        adv = returns - values.detach()

        # ---------- critic update ------------------
        critic_optim.zero_grad()
        value_loss = val_w * (returns - values).pow(2).mean()  # ← has grads
        value_loss.backward()
        critic_optim.step()

        # ---------- actor updates ------------------
        actor_params = [p for ag in agents for p in ag.actor.parameters()]
        actor_optim  = torch.optim.Adam(actor_params, lr=3e-4)

        actor_optim.zero_grad()
        for ag in agents:
            policy_loss  = -(torch.stack(ag.logp) * adv.detach()).mean()
            entropy_loss = -ent_w * torch.stack(ag.ent).mean()
            (policy_loss + entropy_loss).backward()
        actor_optim.step()

        for ag in agents:                          # clear episode traces
            ag.flush()

        # -------- book‑keeping ----------
        episode_rewards.append(sum(global_rewards))
        sharpe_log.append(returns.mean().item())

        if train:
            print(f"Ep {ep+1:>3}  mean‑Sharpe = {sharpe_log[-1]:.4f}")

    if train:
        return np.array(episode_rewards), None, sharpe_log
    else:
        return np.array(episode_rewards), action_logs, sharpe_log

#%% --------------------------------------------------------------------------
# 4.  ──‑‑‑ TRAIN  ‑‑‑———————————————————————————————————————————————————
env = env_train
train_scores, _, sharpe_per_episode = run(episodes=episodes, train=True)

# optional: save checkpoints
for i, ag in enumerate(agents):
    torch.save(ag.actor.state_dict(), f"actor_{i}.pth")
torch.save(critic.state_dict(), "shared_critic.pth")

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
