# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define a simple neural network for approximating the Q-function
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super(DQNetwork, self).__init__()
        # Create a basic feedforward neural network:
        # Input: Flattened window of historical prices
        # Output: Portfolio weights for each stock (via Softmax)
        self.model = nn.Sequential(
            nn.Flatten(),  # flatten the input (e.g., window_size x stock_count)
            nn.Linear(input_dim, hidden_size),  # hidden layer
            nn.ReLU(),  # non-linearity
            nn.Linear(hidden_size, output_dim),  # output layer: one weight per stock
            nn.Softmax(dim=-1)  # ensures output weights sum to 1
        )

    def forward(self, x):
        return self.model(x)  # pass input through the network


# Define the PortfolioAgent using a simple DQN approach
class PortfolioAgent:
    def __init__(
        self, stock_count, window_size=10, lr=1e-3, gamma=0.99,
        epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01
    ):
        self.stock_count = stock_count  # number of stocks this agent manages
        self.input_dim = window_size * stock_count  # flattened input size

        # Main Q-network and target Q-network
        self.model = DQNetwork(self.input_dim, stock_count)
        self.target_model = DQNetwork(self.input_dim, stock_count)
        self.target_model.load_state_dict(self.model.state_dict())  # initialize target model weights

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Discount factor for future rewards
        self.gamma = gamma

        # Epsilon-greedy policy parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Experience replay memory
        self.memory = deque(maxlen=5000)  # capped memory size
        self.batch_size = 32  # training batch size

    def act(self, state):
        """
        Decide portfolio weights for current state.
        Uses epsilon-greedy policy: random or model prediction.
        """
        state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).unsqueeze(0)  # convert to torch tensor and add batch dim

        # With probability epsilon, choose random weights
        if np.random.rand() < self.epsilon:
            weights = np.random.rand(self.stock_count)
            weights /= np.sum(weights)  # normalize to sum to 1
        else:
            with torch.no_grad():
                weights = self.model(state_tensor).numpy().squeeze()  # use model prediction
        return weights

    def update(self, state, action, reward, next_state, done):
        """
        Store experience and perform training using sampled mini-batches.
        """
        # Flatten state and next_state to match input expectations
        self.memory.append((state.flatten(), action, reward, next_state.flatten(), done))

        # Skip training if not enough data yet
        if len(self.memory) < self.batch_size:
            return

        # Sample a random batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)

        # Current predictions (portfolio weights)
        current_weights = self.model(states_tensor)

        # Predicted weights for next states (target network)
        next_weights = self.target_model(next_states_tensor).detach()

        # Construct target values (same shape as current_weights)
        target_weights = current_weights.clone()

        for i in range(self.batch_size):
            target = rewards[i]  # immediate reward
            if not dones[i]:
                # Add discounted future reward estimate
                target += self.gamma * torch.sum(next_weights[i])
            # Distribute total target value uniformly across portfolio weights
            target_weights[i] = target / self.stock_count  # keeps dimension consistent

        # Compute loss between current weights and target weights
        loss = self.loss_fn(current_weights, target_weights)

        # Backpropagation and optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay for exploration-exploitation trade-off
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def sync_target_model(self):
        """
        Periodically sync target network weights from the main model.
        Helps stabilize training (standard DQN technique).
        """
        self.target_model.load_state_dict(self.model.state_dict())
