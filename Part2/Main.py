# main.py
# --------------------------------------------------
#%%
import pandas as pd, numpy as np, torch, time
from Env    import MultiAgentPortfolioEnv
from Agent  import PortfolioAgent
from func   import download_close_prices

#%% --------------------------------------------------------------------------
# 1.  ──‑‑‑ DATA  ‑‑‑——————————————————————————————————————————————————
period_days, num_stocks     = 100, 50
tickers = pd.read_csv(
    "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
).iloc[:num_stocks, 0].tolist()

data = download_close_prices(tickers, start_day="2018-01-01", period_days=365 * 2)
data.dropna(inplace=True)

#%% --------------------------------------------------------------------------
# 2.  ──‑‑‑ ENV + AGENTS  ‑‑‑———————————————————————————————————————————
num_agents, window_size     = 5, 10
stocks_per_agent            = num_stocks // num_agents

env = MultiAgentPortfolioEnv(data, num_agents, window_size)
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
    for ag in agents:
        ag.model.train(mode=train)
        ag.epsilon = 0.0 if not train else ag.epsilon

    t0 = time.perf_counter()      # ➋  start global timer

    for ep in range(1, episodes + 1):
        ep_t0 = time.perf_counter()              # ➌  start episode timer
        state = env.reset()
        done, step, total_r = False, 0, np.zeros(num_agents)

        while not done:
            actions           = [ag.act(st) for ag, st in zip(agents, state)]
            nxt, r, done, _   = env.step(actions)
            total_r          += r
            step             += 1

            if train:
                for i, ag in enumerate(agents):
                    ag.update(state[i], actions[i], r[i], nxt[i], done)
                    if (ep % 2 == 0) and (step == 1):
                        ag.sync_target_model()

            state = nxt

        ep_elapsed = time.perf_counter() - ep_t0 # ➍  episode duration
        print(f"Episode {ep:>3}: Sharpe → {total_r}  "
              f"(took {ep_elapsed:5.2f}s)")

        metrics.append(total_r)

    elapsed = time.perf_counter() - t0          # ➎  total duration
    print(f"\n{'TRAIN' if train else 'EVAL '} finished "
          f"in {elapsed:,.2f} s "
          f"({elapsed/60:.1f} min)")

    return np.vstack(metrics), elapsed

#%% --------------------------------------------------------------------------
# 4.  ──‑‑‑ TRAIN  ‑‑‑———————————————————————————————————————————————————
if __name__ == "__main__":
    train_scores = run(episodes=5, train=True)

    # optional: save checkpoints
    for i, ag in enumerate(agents):
        torch.save(ag.model.state_dict(), f"agent_{i}.pth")

    # ----------------------------------------------------------------------
    # 5.  ──‑‑‑ EVALUATE  ‑‑‑—————————————————————————————————————————————
    # fresh environment for an out‑of‑sample test period if you like
    test_env  = MultiAgentPortfolioEnv(data, num_agents, window_size)
    env = test_env                             # point run() to the test env
    eval_scores = run(episodes=1, train=False)

    print(f"\nEvaluation Sharpe ratios: {eval_scores[-1]}")
