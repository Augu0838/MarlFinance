import numpy as np, torch, torch.nn as nn
from torch.distributions import Dirichlet

# ---------------- shared critic -------------------------------------
class SharedCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):                 # x: (B,state_dim)
        return self.net(x).squeeze(-1)    # (B,)

# ---------------- actor (per agent) ---------------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Softplus()
        )
    def forward(self, obs):
        return self.policy(obs) + 1e-3    # α>0

# --------------- container to hold buffers --------------------------
class AC_Agent:
    """Actor only; critic is shared."""
    def __init__(self, obs_dim, act_dim, actor: Actor):
        self.actor = actor
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # episode buffers
        self.logp, self.ent = [], []

    def act(self, obs):
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        alpha = self.actor(obs_t)
        dist  = Dirichlet(alpha)
        w     = dist.sample()
        self.logp.append(dist.log_prob(w))
        self.ent .append(dist.entropy())
        return w.squeeze(0).detach().numpy()

    def flush(self):
            self.logp.clear(); self.ent.clear()
