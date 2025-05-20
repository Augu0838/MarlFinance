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
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, output_dim)

        # Weight initialization to avoid overconfident logits
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.constant_(self.fc2.bias, 0)

        self.temperature = 2.0  # tune this (1.0–5.0 is common)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x) / self.temperature
        return torch.softmax(logits, dim=-1)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw, scaled logits before softmax (for debugging)"""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.fc2(x) / self.temperature
        return logits


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
    def __init__(self, stock_count, window_size=10, lr=1e-4, gamma=0.99):
        self.stock_count = stock_count
        self.input_dim = window_size * stock_count
        self.gamma = gamma        
        self.device = torch.device("cpu")  
        self.entropy_beta = 0.1  # Entropy regularization coefficient
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        self.actor = DQNetwork(self.input_dim, stock_count).to(self.device)      
        self.critic = ValueNetwork(self.input_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

        self.saved_log_probs = []
        self.entropies = []
        self.rewards = []
        self.states = []  # Save states for critic update

        self.training_logs = {
            "avg_entropy": [],
            "critic_loss": [],
            "actor_loss": [],
            "grad_norms": {
                "fc1.weight": [],
                "fc1.bias": [],
                "fc2.weight": [],
                "fc2.bias": [],
            }
        }

    def act(self, state):
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        probs = self.actor(state_tensor).squeeze()
        probs = torch.clamp(probs, min=1e-3)

        #alpha = probs * 0.05 + 0.001
        alpha = probs * self.stock_count + 1e-2
        alpha = torch.clamp(alpha, min=1e-1) # Ensure alpha is not too small
        
        dist = Dirichlet(alpha)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        #entropy = dist.entropy()
        entropy = -torch.sum(action * torch.log(action + 1e-8))  # ✅ safe entropy estimate

        #print(f"Policy probs: {probs.detach().cpu().numpy().round(2)}")
        # with torch.no_grad():
        #     logits = self.actor.get_logits(state_tensor).squeeze()
        #     print(f"[Raw logits: {logits.cpu().numpy().round(2)}")


        self.saved_log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.states.append(state_tensor)

        return action.detach().cpu().numpy()


    def update(self, episode):
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
            advantages = torch.clamp(advantages, -10.0, 10.0)


        # Actor: train with advantage + entropy bonus
        # Actor loss with clamping
        actor_loss = []
        for log_prob, advantage, entropy in zip(self.saved_log_probs, advantages, self.entropies):
            log_prob = torch.clamp(log_prob, -10.0, 10.0)
            entropy = torch.clamp(entropy, -5.0, 5.0)
            loss = -log_prob * advantage + self.entropy_beta * entropy
            actor_loss.append(loss)

        self.optimizer_actor.zero_grad()
        total_loss = torch.stack(actor_loss).sum()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        #print(f"[EP {episode}] Total grad norm after clipping: {total_norm:.2f}")
        self.optimizer_actor.step()

        self.training_logs["avg_entropy"].append(torch.stack(self.entropies).mean().item())
        self.training_logs["critic_loss"].append(critic_loss.item())
        self.training_logs["actor_loss"].append(total_loss.item())
        for name, param in self.actor.named_parameters():
            if param.grad is not None and name in self.training_logs["grad_norms"]:
                self.training_logs["grad_norms"][name].append(param.grad.norm().item())

        # Clear memory
        self.saved_log_probs.clear()
        self.entropies.clear()
        self.rewards.clear()
        self.states.clear()