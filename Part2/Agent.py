# agent.py

import numpy as np

class PortfolioAgent:
    """
    Simple agent placeholder. For real training, integrate with a DRL algorithm.
    """
    def __init__(self, stock_count, random_seed=None):
        self.stock_count = stock_count
        if random_seed is not None:
            np.random.seed(random_seed)

    def act(self, state):
        """
        Decide portfolio weights given the state (price history).
        Here: uniform random weights that sum to 1.
        """
        weights = np.random.rand(self.stock_count)
        weights /= np.sum(weights)
        return weights

    def update(self, state, action, reward, next_state, done):
        """
        Placeholder for learning step. Not used in this stub.
        """
        pass
