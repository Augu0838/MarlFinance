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

# ---------------------------------------------------------------------------
# 1. PolicyNetwork (actor) – NOT a Q‑network
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """A two‑layer MLP that outputs a **probability distribution** over stocks.

    Parameters
    ----------
    input_dim : int
        Length of the flattened state vector
        (= ``window_size × stock_count``).
    output_dim : int
        Number of stocks → size of the weight vector.
    hidden_size : int, default 128
        Width of the single hidden layer.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128):
        super().__init__()

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


# ---------------------------------------------------------------------------
# 2. PortfolioAgent – REINFORCE with entropy regularisation
# ---------------------------------------------------------------------------
class PortfolioAgent:
    """Policy‑gradient agent that learns to allocate capital across *k* stocks.

    Workflow per episode
    --------------------
    1. **act()** – sample a weight vector, store log‑prob & entropy.
    2. Environment returns a *Sharpe ratio* reward → store it.
    3. After the episode ends, **update()**:
       * builds discounted return ``R_t`` for every step;
       * computes ``loss = −logπ(a_t|s_t) * R_t − λ·entropy``;
       * back‑props and updates the network.

    Notes
    -----
    * Uses **Dirichlet(α)** so sampled actions always lie on the simplex.
    * Multiplying the policy output by 5 makes α > 1, giving *moderate* noise.
    * ``entropy_weight`` controls exploration strength (λ).
    """

    # ------------------------------------------------------------------ init
    def __init__(self, stock_count: int, window_size: int = 10,
                 lr: float = 1e-3, gamma: float = 0.99):
        # ----- hyper‑parameters
        self.stock_count = stock_count
        self.input_dim   = window_size * stock_count
        self.gamma       = gamma        # discount factor

        # ----- neural network + optimiser
        self.model = PolicyNetwork(self.input_dim, stock_count)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # ----- episode memory (cleared every update)
        self.saved_log_probs: list[torch.Tensor] = []  # log π(a|s)
        self.rewards:         list[float]        = []  # r_t
        self.entropies:       list[torch.Tensor] = []  # H[π]

    # ------------------------------------------------------------------ act()
    def act(self, state: np.ndarray) -> np.ndarray:
        """Return a legal weight vector and log info for learning."""
        # 1) convert state → torch (shape: (1, input_dim))
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)

        # 2) network preference (probabilities sum to 1)
        probs = self.model(state_tensor).squeeze()

        # 3) Dirichlet exploration around `probs`
        alpha  = probs * 5 + 1e-3  # α > 1 ⇒ mild noise; tweak multiplier as needed
        dist   = Dirichlet(alpha)
        action = dist.sample()      # sampled weights (positive, sum‑to‑1)

        # 4) bookkeeping for REINFORCE
        self.saved_log_probs.append(dist.log_prob(action))  # ln π(a|s)
        self.entropies.append(dist.entropy())               # H[π]

        return action.detach().numpy()  # ← environment expects NumPy array

    # -------------------------------------------------------------- update()
    def update(self) -> None:
        """Perform one REINFORCE update using data from the last episode."""
        # 1) Compute discounted returns G_t
        R = 0.0
        returns: list[float] = []
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)  # prepend so list ends in time order
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # 2) Normalise returns (helps gradient stability)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

        # 3) Policy loss = −logπ(a|s)·G_t  − λ·entropy
        entropy_weight = 0.01
        policy_losses = [-(lp * G) - entropy_weight * H
                         for lp, H, G in zip(self.saved_log_probs,
                                             self.entropies,
                                             returns_t)]

        # 4) Gradient descent step
        self.optimizer.zero_grad()
        torch.stack(policy_losses).sum().backward()
        self.optimizer.step()

        # 5) Clear buffers for next episode
        self.saved_log_probs.clear()
        self.rewards.clear()
        self.entropies.clear()
