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

class PortfolioAgent:
    def __init__(self, stock_count, window_size=10, lr=1e-3, gamma=0.99, alpha_scale=0.25):
        self.stock_count = stock_count
        self.input_dim = window_size * stock_count
        self.gamma = gamma
        self.alpha_scale = alpha_scale
        
        self.device = torch.device("cpu")  
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        self.actor = DQNetwork(self.input_dim, stock_count).to(self.device)      
        self.critic = ValueNetwork(self.input_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.saved_log_probs = []
        self.rewards = []
        self.states = []  # Save states for critic update

    def act(self, state):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.actor(state_tensor).squeeze()

        alpha = probs * self.alpha_scale + 1e-2
        dist = Dirichlet(alpha)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.saved_log_probs.append(log_prob)
        self.states.append(state_tensor)

        return action.detach().cpu().numpy()
    
    def update_single(self):
        if len(self.rewards) < 1:
            return

    # Prepare single step
        R = self.rewards[-1]
        state_tensor = self.states[-1]  # shape: [1, input_dim]
        return_tensor = torch.tensor([R], dtype=torch.float32, device=self.device)  # shape: [1]

        # Critic: predict scalar value
        value = self.critic(state_tensor).squeeze()  # ensures shape is [] (scalar) or [1]

        # Match shapes explicitly for MSELoss
        value = value.view(-1)           # shape: [1]
        return_tensor = return_tensor.view(-1)  # ensure same shape

        # Compute critic loss
        critic_loss = nn.MSELoss()(value, return_tensor)

        # Update critic
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Actor update
        with torch.no_grad():
            advantage = return_tensor - value.detach()

        actor_loss = -self.saved_log_probs[-1] * advantage

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Clear only most recent memory
        self.rewards.pop()
        self.saved_log_probs.pop()
        self.states.pop()


    def update(self):
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Critic: train to predict returns
        state_batch = torch.cat(self.states).to(self.device)
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
