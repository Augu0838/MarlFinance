# env_fast.py ---------------------------------------------------------------
import numpy as np
import gym
from gym import spaces

class MultiAgentPortfolioEnvFast(gym.Env):
    """
    Same interface, but all computations are vectorised numpy.
    """
    def __init__(self, stock_df, num_agents, window_size=10):
        super().__init__()
        self.window_size   = window_size
        self.num_agents    = num_agents

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
        window_ret = window_ret.reshape(self.window_size-1,
                                         self.num_agents,
                                         self.stocks_per_agent)

        # (b) stack actions -> shape (num_agents, stocks_per_agent)
        A = np.vstack(actions).astype(np.float32)

        # (c) portfolio returns per agent in one mat‑mul
        #     window_ret: (W-1, A, K) , A.T: (K, A) → (W-1, A)
        port_ret = np.einsum("wak,ak->wa", window_ret, A)   # incredibly fast

        mean  = port_ret.mean(axis=0)                       # (A,)
        std   = port_ret.std(axis=0)  + 1e-6
        rewards = (mean / std).tolist()                     # python list OK

        self.current_step += 1
        done = self.current_step >= self.num_steps
        return self._get_obs(), rewards, done, {}
