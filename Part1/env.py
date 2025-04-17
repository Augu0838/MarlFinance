import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A simple trading environment for a single stock.
    States: market data for a given day (here, we assume one feature: price)
    Actions: 0 = hold, 1 = buy, 2 = sell
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)  # assume df rows are ordered by time
        self.n_days = len(self.df)
        self.current_day = None
        
        # Define the observation space.
        # In this simple example, the observation is just the stock price (or can be expanded)
        # Using a Box space: one-dimensional continuous value.
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.df.shape[1],), dtype=np.float32)
        
        # Define the action space: 0 = hold, 1 = buy, 2 = sell.
        self.action_space = spaces.Discrete(3)
        
        # Portfolio parameters
        self.initial_cash = 10000.0
        self.cash = None
        self.stock_owned = None
        self.prev_portfolio_value = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.cash = self.initial_cash
        self.stock_owned = 0
        # Initial portfolio value: cash only
        self.prev_portfolio_value = self.initial_cash
        # Return the initial observation
        return self._get_obs(), {}
    
    def step(self, action):
        # Get current price
        price = float(self.df.iloc[self.current_day]['Price'])
        
        # Execute action: 0 = hold, 1 = buy, 2 = sell
        if action == 1:  # Buy one share if enough cash
            if self.cash >= price:
                self.stock_owned += 1
                self.cash -= price
        elif action == 2:  # Sell one share if any owned
            if self.stock_owned > 0:
                self.stock_owned -= 1
                self.cash += price
        # If action is 0, we hold
        
        # Calculate new portfolio value and reward as change from previous day
        portfolio_value = self.cash + self.stock_owned * price
        reward = portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value
        
        # Move to the next day
        self.current_day += 1
        done = self.current_day >= self.n_days
        
        info = {'portfolio_value': portfolio_value, 'cash': self.cash, 'stock_owned': self.stock_owned}
        
        # If we've reached the end, return the last available observation
        if done:
            obs = self.df.iloc[-1].values.astype(np.float32)
        else:
            obs = self._get_obs()
        
        return obs, reward, done, False, info


    def _get_obs(self):
        # Return current market data as observation.
        # Here, we simply return the row values of the DataFrame for the current day.
        # Make sure to convert to np.float32.
        return self.df.iloc[self.current_day].values.astype(np.float32)
    
    def render(self, mode='human'):
        price = float(self.df.iloc[self.current_day-1]['Price'])
        portfolio_value = self.cash + self.stock_owned * price
        print(f"Day {self.current_day}: Price: {price:.2f}, Cash: {self.cash:.2f}, "
              f"Stock Owned: {self.stock_owned}, Portfolio Value: {portfolio_value:.2f}")
    
    def close(self):
        pass

# -------------------------------
# Example usage:
# -------------------------------
if __name__ == '__main__':
    # Create a dummy DataFrame for demonstration.
    # In practice, replace this with your actual trading data.
    dates = pd.date_range('2018-01-01', periods=200, freq='B')
    prices = np.linspace(100, 200, num=200)  # price increases linearly from 100 to 200
    df = pd.DataFrame({'Price': prices}, index=dates)
    
    # Create the environment
    env = TradingEnv(df)
    
    # Example run: train (or test) using a simple random policy
    episodes = 5
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            # Replace with your policy. Here we sample randomly.
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
        print(f"Episode {ep+1} ended. Total Reward: {total_reward:.2f}")
    env.close()
