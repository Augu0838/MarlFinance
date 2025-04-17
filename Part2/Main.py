# main.py

import pandas as pd
import numpy as np
from Env import MultiAgentPortfolioEnv
from Agent import PortfolioAgent

# Generate mock stock data (500 stocks, 1000 days)
dates = pd.date_range(start="2010-01-01", periods=1000, freq='D')
stock_names = [f'STK{i}' for i in range(50)]
mock_data = pd.DataFrame(np.random.rand(1000, 50) * 100, index=dates, columns=stock_names)

# Initialize environment and agents
num_agents = 5
env = MultiAgentPortfolioEnv(stock_data=mock_data, num_agents=num_agents, window_size=10)
agents = [PortfolioAgent(stock_count=10, random_seed=i) for i in range(num_agents)]  # 500 / 5 = 100 per agent

# Training loop (simple rollout, no learning yet)
episodes = 10
for ep in range(episodes):
    print(f"Episode {ep + 1}")
    states = env.reset()
    done = False
    total_rewards = np.zeros(num_agents)

    while not done:
        actions = [agent.act(state) for agent, state in zip(agents, states)]
        next_states, rewards, env_done, _ = env.step(actions)
        done = env_done
        for i, agent in enumerate(agents):
            agent.update(states[i], actions[i], rewards[i], next_states[i], done)
            total_rewards[i] += rewards[i]
        states = next_states

    print(f"Total Sharpe Ratios (shared): {total_rewards}")
