"""agent.py
-------------------------------------------------------------------------------
A *policy‑gradient* agent for portfolio optimisation.

Why this file exists
--------------------
*   The **environment** shows the agent a small window of recent prices.
*   The **agent** must output a *vector of portfolio weights* (one per stock)
    that always sums to **1**.
*   Classic DQN works poorly for that continuous, simplex‑constrained action
    space, so we use **REINFORCE** with a Dirichlet sampler.

Core ideas implemented here
---------------------------
* **PolicyNetwork** – maps state → preferred allocation probabilities.
* **Dirichlet exploration** – keeps each sampled allocation legal (positive &
  summing to 1) while adding controllable randomness.
* **Entropy bonus** – prevents the policy from collapsing too early.
* **Discounted‑return baseline** – each action is credited according to the
  *future* Sharpe ratios it eventually produced.
"""

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet

# Define a simple neural network for approximating the Q-function
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(DQNetwork, self).__init__()
        # Create a basic feedforward neural network:
        # Input: Flattened window of historical prices
        # Output: Portfolio weights for each stock (via Softmax)
        self.model = nn.Sequential(
            nn.Flatten(),                      # (B, window, k) -> (B, window*k)
            nn.Linear(input_dim, hidden_size), # dense layer
            nn.ReLU(),                         # non‑linearity
            nn.Linear(hidden_size, output_dim),
            nn.Softmax(dim=-1)                 # convert logits → probabilities
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return a **(batch, output_dim)** tensor of probabilities."""
        return self.model(x)

# Class for the critict
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Output a single value estimate
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)  # (batch,) shape

# ---------------------------------------------------------------------------
# 2. PortfolioAgent – REINFORCE with entropy regularisation
# ---------------------------------------------------------------------------
# class PortfolioAgent:
#     def __init__(
#         self, stock_count, window_size=10, lr=1e-3, gamma=0.99):
#         self.stock_count = stock_count  # number of stocks this agent manages
#         self.input_dim = window_size * stock_count  # flattened input size
#         self.entropies = []  # store entropy values per step

#         # Main Q-network
#         self.model = DQNetwork(self.input_dim, stock_count)

#         # Optimizer
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

#         # Discount factor for future rewards
#         self.gamma = gamma

#         # Stores for one episode
#         self.saved_log_probs = []
#         self.rewards = []
class PortfolioAgent:
    def __init__(self, stock_count, window_size=10, lr=1e-3, gamma=0.99):
        self.stock_count = stock_count
        self.input_dim = window_size * stock_count
        self.gamma = gamma
        
        self.actor = DQNetwork(self.input_dim, stock_count)
        self.critic = ValueNetwork(self.input_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []  # Save states for critic update


    # def act(self, state):
    #     """
    #     Sample portfolio weights from a Dirichlet distribution centered on the model output.
    #     Save log-prob for policy gradient training.
    #     """
    #     state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)  # convert to torch tensor and add batch dim
    #     probs = self.model(state_tensor).squeeze()

    #     # Add temperature for exploration; adjust concentration (alpha)
    #     alpha = probs * 0.05 + 1e-3  # scale to get sharper distributions
    #     dist = Dirichlet(alpha)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     entropy = dist.entropy()

    #     self.saved_log_probs.append(log_prob)
    #     self.entropies.append(entropy)

    #     return action.detach().numpy()

    def act(self, state):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze()

        alpha = probs * 0.05 + 1e-3
        dist = Dirichlet(alpha)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.saved_log_probs.append(log_prob)
        self.states.append(state_tensor)

        return action.detach().numpy()

    # def update(self):
    #     """
    #     REINFORCE with entropy regularization.
    #     """
    #     R = 0
    #     returns = []
    #     for r in reversed(self.rewards):
    #         R = r + self.gamma * R
    #         returns.insert(0, R)

    #     returns = torch.tensor(returns, dtype=torch.float32)
    #     returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    #     entropy_weight = 0.05  # you can tune this
    #     policy_loss = []

    #     for log_prob, entropy, R in zip(self.saved_log_probs, self.entropies, returns):
    #         loss = -log_prob * R - entropy_weight * entropy
    #         policy_loss.append(loss)

    #     self.optimizer.zero_grad()
    #     torch.stack(policy_loss).sum().backward()
    #     self.optimizer.step()

    #     # Clear episode history
    #     self.saved_log_probs.clear()
    #     self.rewards.clear()
    #     self.entropies.clear()
    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Critic: train to predict returns
        state_batch = torch.cat(self.states)
        values = self.critic(state_batch)
        critic_loss = nn.MSELoss()(values, returns)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor: train with advantage
        with torch.no_grad():
            advantages = returns - self.critic(state_batch)

        actor_loss = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            actor_loss.append(-log_prob * advantage)

        self.optimizer_actor.zero_grad()
        torch.stack(actor_loss).sum().backward()
        self.optimizer_actor.step()

        # Clear memory
        self.saved_log_probs.clear()
        self.rewards.clear()
        self.states.clear()
