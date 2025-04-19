# env_fast.py ---------------------------------------------------------------
import numpy as np
import gym
from gym import spaces
import pandas as pd

class MultiAgentPortfolioEnv(gym.Env):
    """
    Same interface, but all computations are vectorised numpy.
    """
    def __init__(self, stock_df, num_agents, window_size=10, external_trader=None):
        super().__init__()
        self.window_size   = window_size
        self.num_agents    = num_agents
        self.external_trader = external_trader # accept input from external trader

        # --- 1)  cache ndarray & returns ----------------------------------
        self.prices   = stock_df.to_numpy(dtype=np.float32)          # shape (T, S)
        self.returns  = np.diff(self.prices, axis=0) / self.prices[:-1]  # (T-1, S)

        self.num_steps, self.num_stocks = self.prices.shape
        self.stocks_per_agent = self.num_stocks // num_agents

        # --- 2)  gym spaces ------------------------------------------------
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(window_size, self.stocks_per_agent), dtype=np.float32
        )
        self.action_space      = spaces.Box(
            0, 1, shape=(self.stocks_per_agent,), dtype=np.float32
        )

        self.current_step = window_size

    # ----------------------------------------------------------------------
    def reset(self):
        self.current_step = self.window_size
        return self._get_obs()

    # ----------------------------------------------------------------------
    def _get_obs(self):
        """
        Returns a list with len=num_agents, each of shape
        (window_size, stocks_per_agent)
        No Python loops: one reshape + one slice.
        """
        start = self.current_step - self.window_size
        end   = self.current_step

        window = self.prices[start:end]                      # (W, S)
        window = window.reshape(self.window_size,
                                self.num_agents,
                                self.stocks_per_agent)
        return [window[:, i, :] for i in range(self.num_agents)]

    # ----------------------------------------------------------------------
    def step(self, actions):
        """
        actions  : list(np.ndarray) length = num_agents
        Vectorised Sharpe ratio across all agents in one shot.
        """
        # (a) slice the correct rows from the return matrix
        start = self.current_step - self.window_size
        end   = self.current_step - 1                       # returns are T‑1

        window_ret = self.returns[start:end]                # (W‑1, S)

        # Combine all agent actions into one portfolio
        agent_portfolio = np.vstack(actions).astype(np.float32).flatten()  # (S,)

        # Include external trader if provided
        if self.external_trader:
            window_df = pd.DataFrame(
                self.prices[self.current_step - self.window_size: self.current_step],
                columns=[f"Stock{i}" for i in range(self.num_stocks)]
            )
            momentum_weights = self.external_trader.step(window_df)  # shape (S,)
            combined_portfolio = agent_portfolio + momentum_weights
            combined_portfolio /= combined_portfolio.sum()  # re-normalize
        else:
            combined_portfolio = agent_portfolio

        # Portfolio returns over time
        port_ret = np.dot(window_ret, combined_portfolio)             # (W-1,)

        mean  = port_ret.mean(axis=0)                       # (A,)
        std   = port_ret.std(axis=0)  + 1e-6

        reward = mean / std
        rewards = [reward] * self.num_agents

        self.current_step += 1
        done = self.current_step >= self.num_steps
        return self._get_obs(), rewards, done, {}
