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
        self.stock_df = stock_df

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
    # def _get_obs(self):
    #     """
    #     Returns a list with len=num_agents, each of shape
    #     (window_size, stocks_per_agent)
    #     No Python loops: one reshape + one slice.
    #     """
    #     start = self.current_step - self.window_size
    #     end   = self.current_step

    #     window = self.prices[start:end]                      # (W, S)
    #     window = window.reshape(self.window_size,
    #                             self.num_agents,
    #                             self.stocks_per_agent)
        
    #     return [window[:, i, :] for i in range(self.num_agents)]
    # def _get_obs(self):
    #     start = self.current_step - self.window_size
    #     end = self.current_step

    #     window = self.prices[start:end]  # (W, S)
    #     window = window.reshape(self.window_size, self.num_agents, self.stocks_per_agent)
    #     return [window[:, i, :] for i in range(self.num_agents)]
    
    # get obs with covariance returns
    def _get_obs(self):
        """
        Constructs per-agent observations, including price window and correlation
        to external portfolio for each managed stock.
        Returns a list of observations, one per agent.
        """
        start = self.current_step - self.window_size
        end = self.current_step

        window_prices = self.prices[start:end]  # shape: (w, S)
        returns = np.diff(np.log(window_prices), axis=0)  # shape: (w-1, S)

        # Get external weights
        date = self.stock_df.index[self.current_step]
        if self.external_trader is not None and date in self.external_trader.index:
            external_weights = self.external_trader.loc[date].values.astype(np.float32)
        else:
            external_weights = np.ones(self.num_stocks, dtype=np.float32) / self.num_stocks

        external_ret = np.dot(returns, external_weights)  # shape: (w-1,)

        # Correlation of each stock with external portfolio
        correlations = []
        for i in range(self.num_stocks):
            stock_ret = returns[:, i]
            corr = np.corrcoef(stock_ret, external_ret)[0, 1]
            correlations.append(np.nan_to_num(corr, nan=0.0))

        correlations = np.array(correlations)  # shape: (S,)
        observations = []

        for agent_id in range(self.num_agents):
            start_idx = agent_id * self.stocks_per_agent
            end_idx = start_idx + self.stocks_per_agent

            price_window = window_prices[:, start_idx:end_idx]  # shape: (w, k)
            agent_corr = correlations[start_idx:end_idx]
            corr_matrix = np.tile(agent_corr, (self.window_size, 1))  # shape: (w, k)

            # Stack price and correlation as 2 channels
            obs = np.stack([price_window, corr_matrix], axis=0)  # shape: (2, w, k)
            observations.append(obs)

        return observations

    # ----------------------------------------------------------------------
    # def step(self, actions):
    #     """
    #     actions  : list(np.ndarray) length = num_agents
    #     Vectorised Sharpe ratio across all agents in one shot.
    #     """
    #     # (a) slice the correct rows from the return matrix
    #     start = self.current_step - self.window_size
    #     end   = self.current_step - 1                       # returns are T‑1

    #     window_ret = self.returns[start:end]                # (W‑1, S)

    #     # Combine all agent actions into one portfolio
    #     agent_portfolio = np.vstack(actions).astype(np.float32).flatten()  # (S,)

    #     # Combine with external weights if available
    #     if self.external_trader is not None:
    #         date = self.stock_df.index[self.current_step]
    #         if date in self.external_trader.index:
    #             external_weights = self.external_trader.loc[date].values.astype(np.float32)
    #             combined_portfolio = agent_portfolio + external_weights
    #             combined_portfolio /= combined_portfolio.sum()  # re-normalize
    #         else:
    #             combined_portfolio = agent_portfolio  # fallback
    #     else:
    #         combined_portfolio = agent_portfolio

    #     # Portfolio returns over time
    #     port_ret = np.dot(window_ret, combined_portfolio)             # (W-1,)

    #     mean  = port_ret.mean(axis=0)                       # (A,)
    #     std   = port_ret.std(axis=0)  + 1e-6
    #     reward = mean / std

    #     rewards = [reward] * self.num_agents

    #     self.current_step += 1
    #     done = self.current_step >= self.num_steps
    #     return self._get_obs(), rewards, done, {}

    # Step function with covariances
    def step(self, actions):
        start = self.current_step - self.window_size
        end = self.current_step - 1

        window_ret = self.returns[start:end]  # shape: (w-1, S)
        agent_portfolio = np.vstack(actions).astype(np.float32).flatten()

        date = self.stock_df.index[self.current_step]
        if self.external_trader is not None and date in self.external_trader.index:
            external_weights = self.external_trader.loc[date].values.astype(np.float32)
            combined_portfolio = agent_portfolio #+ external_weights
            combined_portfolio /= combined_portfolio.sum()  # re-normalize
        else:
            combined_portfolio = agent_portfolio 

        port_ret = np.dot(window_ret, combined_portfolio)
        mean = port_ret.mean()
        std = port_ret.std() + 1e-6
        reward = mean / std

        rewards = [reward] * self.num_agents

        # Generate next observation before advancing the step counter
        obs = self._get_obs()

        self.current_step += 1
        done = self.current_step >= self.num_steps

        return obs, rewards, done, {}


