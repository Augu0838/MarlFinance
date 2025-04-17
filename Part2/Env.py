import numpy as np
import pandas as pd
import gym
from gym import spaces

class MultiAgentPortfolioEnv(gym.Env):
    """
    Multi-agent environment for managing a portfolio of S&P 500 stocks using the Sharpe Ratio.
    Each agent controls a subset of stocks. The Sharpe Ratio is computed across all agents' portfolios.
    """

    def __init__(self, stock_data: pd.DataFrame, num_agents: int, window_size: int = 10):
        super(MultiAgentPortfolioEnv, self).__init__()
        self.stock_data = stock_data  # DataFrame with MultiIndex (date, stock)
        self.num_agents = num_agents
        self.window_size = window_size
        self.num_stocks = len(stock_data.columns)
        self.stocks_per_agent = self.num_stocks // num_agents

        # Define observation and action space per agent
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, self.stocks_per_agent), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.stocks_per_agent,), dtype=np.float32)

        self.current_step = self.window_size
        self.done = False

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        return self._get_observations()

    def _get_observations(self):
        # Return a list of observations, one per agent
        obs = []
        for i in range(self.num_agents):
            start_idx = i * self.stocks_per_agent
            end_idx = (i + 1) * self.stocks_per_agent
            sub_data = self.stock_data.iloc[self.current_step - self.window_size:self.current_step, start_idx:end_idx]
            obs.append(sub_data.values)
        return obs

    def step(self, actions):
        """
        actions: list of np.array, one per agent, representing portfolio weights
        """
        assert len(actions) == self.num_agents
        rewards = []

        # Calculate portfolio returns per agent
        for i, action in enumerate(actions):
            start_idx = i * self.stocks_per_agent
            end_idx = (i + 1) * self.stocks_per_agent
            prices = self.stock_data.iloc[self.current_step - self.window_size:self.current_step, start_idx:end_idx]
            returns = prices.pct_change().dropna().values
            portfolio_returns = np.dot(returns, action)
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-6)
            rewards.append(sharpe_ratio)

        # Total Sharpe Ratio across agents
        total_reward = np.mean(rewards)

        # Broadcast total Sharpe Ratio as reward to all agents
        rewards = [total_reward] * self.num_agents

        self.current_step += 1
        if self.current_step >= len(self.stock_data):
            self.done = True

        return self._get_observations(), rewards, self.done, {}

    def render(self, mode='human'):
        pass  # Could be extended to show portfolio values over time
