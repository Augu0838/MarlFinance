import gym
from gym import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    def __init__(self, data, initial_capital=1e6, transaction_cost=0.001):
        super(StockTradingEnv, self).__init__()
        
        # Each state is just the current time index.
        self.data = data
        self.n_steps = len(data)
        self.tickers = self.data.columns
        self.n_stocks = len(self.tickers)
        
        # Action space: 3 possibilities (Buy, Hold, Sell) for each stock
        # => total 3 * n_stocks discrete actions
        self.action_space = spaces.Discrete(self.n_stocks * 3)
        
        # Observation space: a single integer index from 0..(n_steps-1)
        # self.observation_space = spaces.Discrete(self.n_steps)

        # Example: state = [ day_index, holdings..., cash ]
        # Alternatively could also include current prices, returns, etc.
        # For demonstration, shape = (n_stocks + 2,)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.n_stocks + 2,), dtype=np.float32
        )
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Internal state
        self.reset()

    def _get_prices(self, step):
        """
        Returns the current (or next) prices for each ticker at 'step'.
        """
        return self.data.iloc[step].values  # shape = (n_stocks,)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.cash = self.initial_capital
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        # Example observation: [ day_index, holdings..., cash ]
        obs = np.concatenate(([self.current_step], self.holdings, [self.cash]))
        return obs

    def step(self, action):
        # Decode action: action in [0 .. 3*n_stocks-1].
        # Each ticker has 3 possibilities: (0=Sell, 1=Hold, 2=Buy).
        # E.g. if n_stocks=3, action=5 => means for ticker0=Hold, ticker1=Buy, ticker2=Sell (just an example).
        # One way is to decode each ticker's sub-action by dividing by 3, etc.
        sub_actions = []
        tmp = action
        for _ in range(self.n_stocks):
            sub_actions.append(tmp % 3)
            tmp //= 3
        sub_actions = sub_actions[::-1]  # reverse to match ticker0..n-1 if desired

        # Get current prices
        current_prices = self._get_prices(self.current_step)
        
        # We will compute the old portfolio value
        old_portfolio_val = self._get_portfolio_value(current_prices)

        # Apply each sub_action: 0=Sell 1=Hold 2=Buy
        for i, sub_a in enumerate(sub_actions):
            if sub_a == 2:
                # Buy 1 share of ticker i
                cost = current_prices[i] * (1 + self.transaction_cost)
                if cost <= self.cash:
                    self.holdings[i] += 1
                    self.cash -= cost
            elif sub_a == 0:
                # Sell 1 share (if we have it)
                if self.holdings[i] > 0:
                    self.holdings[i] -= 1
                    proceeds = current_prices[i] * (1 - self.transaction_cost)
                    self.cash += proceeds
            # If sub_a == 1 => hold, do nothing

        # Advance time
        self.current_step += 1
        terminated = (self.current_step >= self.n_steps - 1)
        
        # Compute new portfolio value after action
        new_prices = self._get_prices(self.current_step)
        new_portfolio_val = self._get_portfolio_value(new_prices)

        # Reward = daily change in portfolio (could incorporate risk, but keep it simple for Q-learning)
        reward = new_portfolio_val - old_portfolio_val

        # Return the new observation (Box), along with reward, done flags, etc.
        obs = self._get_observation()  # e.g. a NumPy array
        return obs, reward, terminated, {}

    def _get_portfolio_value(self, prices):
        return np.sum(self.holdings * prices) + self.cash

    def close(self):
        pass
