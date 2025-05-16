import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from Agent import PortfolioAgent  # Your custom agent file
from Env import MultiAgentPortfolioEnv
from Main import run, env_train, env_test, test_data, external_trader  # import from your main module

# Set parameters
alpha_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
episodes = 50
window_size = 20
num_agents = 1
stocks_per_agent = 98 * 5
device = torch.device("cpu")

# Store results
sharpe_results = []

for alpha in alpha_values:
    print(f"\nTesting alpha_scale = {alpha}")

    # Re-initialize environment and agents
    agents = [PortfolioAgent(stock_count=stocks_per_agent, window_size=window_size, alpha_scale=alpha)
              for _ in range(num_agents)]
    
    # Assign agents globally for use in run()
    globals()["agents"] = agents
    globals()["env"] = env_train

    # Train
    train_scores, _, _ = run(episodes=episodes, train=True)

    # Save model
    for i, ag in enumerate(agents):
        torch.save({
            'actor_state_dict': ag.actor.state_dict(),
            'critic_state_dict': ag.critic.state_dict()
        }, f"agent_{i}_checkpoint_alpha_{alpha}.pth")

    # Load models and evaluate
    for i, ag in enumerate(agents):
        checkpoint = torch.load(f"agent_{i}_checkpoint_alpha_{alpha}.pth", map_location=device)
        ag.actor.load_state_dict(checkpoint['actor_state_dict'])
        ag.critic.load_state_dict(checkpoint['critic_state_dict'])

    env = env_test
    eval_scores, _, action_logs = run(episodes=1, train=False)

    # Calculate combined Sharpe
    returns = np.diff(test_data.values, axis=0) / test_data.values[:-1]
    dates = test_data.index[1:]
    start_idx = env.window_size
    max_len = min(len(action_logs[0]), len(dates) - start_idx)
    eval_dates = dates[start_idx : start_idx + max_len]
    ret_window = returns[start_idx : start_idx + max_len]

    combined_daily_returns = []
    for t in range(max_len):
        step_actions = action_logs[0][t]
        agent_weights = np.vstack(step_actions).flatten()
        date = eval_dates[t]
        if date in external_trader.index:
            ext_weights = external_trader.loc[date].values
            combo_weights = agent_weights + ext_weights
            combo_weights /= combo_weights.sum()
        else:
            combo_weights = agent_weights
        r = np.dot(ret_window[t], combo_weights)
        combined_daily_returns.append(r)

    def rolling_sharpe(returns, window=window_size):
        r = np.array(returns)
        mean = pd.Series(r).rolling(window).mean()
        std = pd.Series(r).rolling(window).std() + 1e-6
        return (mean / std).values

    sharpe_combined = np.nan_to_num(rolling_sharpe(combined_daily_returns), nan=0.0)
    mean_sharpe_combined = np.mean(sharpe_combined)
    sharpe_results.append(mean_sharpe_combined)

    print(f"Alpha {alpha} → Mean Sharpe: {mean_sharpe_combined:.4f}")

# Plot results
plt.figure()
plt.plot(alpha_values, sharpe_results, marker='o')
plt.xlabel("Alpha Scale")
plt.ylabel("Mean Sharpe Ratio")
plt.title("Alpha Grid Search – Sharpe Ratio")
plt.grid(True)
plt.show()

# Print summary
for alpha, sharpe in zip(alpha_values, sharpe_results):
    print(f"Alpha {alpha:.2f} → Sharpe: {sharpe:.4f}")

best_alpha = alpha_values[np.argmax(sharpe_results)]
print(f"\nBest alpha_scale: {best_alpha}")
