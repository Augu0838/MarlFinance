import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm, skew, kurtosis

# ------------------ Sharpe by episode ------------------
def plot_training_sharpe(sharpe_series):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(sharpe_series) + 1), sharpe_series, marker='')
    plt.title("Average Sharpe Ratio per Training Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Sharpe ratio ------------------
def sharpe_ratios(sharpe_combined, sharpe_external, test_data, title='Rolling Sharpe Ratio (20-day)', x_title='Date'):
    # Normalize prices
    normalized_prices = test_data / test_data.iloc[0] * 100
    daily_returns = normalized_prices.pct_change().dropna()

    # Market rolling Sharpe ratio
    rolling_mean = daily_returns.mean(axis=1).rolling(window=20).mean()
    rolling_std  = daily_returns.mean(axis=1).rolling(window=20).std()
    sharpe_rolling = rolling_mean / (rolling_std + 1e-8)


    # Align on shared date index
    common_idx = sharpe_combined.index.intersection(sharpe_external.index).intersection(sharpe_rolling.index)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(common_idx, sharpe_combined.loc[common_idx], label='Combined Portfolio')
    plt.plot(common_idx, sharpe_external.loc[common_idx], label='External-only Portfolio')
    plt.plot(common_idx, sharpe_rolling.loc[common_idx], label='Market Sharpe Ratio', linestyle='--', color='gray')

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------ Sharpe difference ------------------
def sharp_difference(sharpe_combined, sharpe_external, title='Sharpe Ratio Difference'):
    # Compute difference between the rolling Sharpe ratios
    sharpe_diff = np.array(sharpe_combined) - np.array(sharpe_external)
    days = np.arange(len(sharpe_combined))

    # Plot the difference
    plt.figure(figsize=(10, 5))
    plt.plot(days, sharpe_diff, label='Sharpe Ratio Difference (Combined - External)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Sharpe Difference')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Cumulative Returns ------------------
def cumulative_returns(dates, combined_daily_returns, external_daily_returns):

    # Number of initial days to skip and rebased to 1 at that point
    offset=0

    combined_cum_returns = np.cumprod([1 + r for r in combined_daily_returns])
    external_cum_returns = np.cumprod([1 + r for r in external_daily_returns])

    # Rebase both to start at 1 from `offset`
    combined_cum_rebased = combined_cum_returns[offset:] / combined_cum_returns[offset]
    external_cum_rebased = external_cum_returns[offset:] / external_cum_returns[offset]

    # Align dates
    dates_array = np.array(dates[offset:offset + len(combined_cum_rebased)])

    plt.figure(figsize=(10, 5))
    plt.plot(dates_array[:len(combined_cum_rebased)], combined_cum_rebased, label='Combined Portfolio')
    plt.plot(dates_array[:len(external_cum_rebased)], external_cum_rebased, label='External-only Portfolio')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Histograms with Stats ------------------
def histogram(combined_daily_returns, external_daily_returns):
    # Trim initial period
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Combined portfolio
    mu_comb = np.mean(combined_daily_returns)
    sigma_comb = np.std(combined_daily_returns)
    skew_comb = skew(combined_daily_returns)
    kurt_comb = kurtosis(combined_daily_returns)
    print(len(combined_daily_returns))
    # Histogram
    axs[0].hist(combined_daily_returns, bins=50, alpha=0.7, color='blue', density=True)

    # Normal curve
    x_comb = np.linspace(mu_comb - 4*sigma_comb, mu_comb + 4*sigma_comb, 500)
    axs[0].plot(x_comb, norm.pdf(x_comb, mu_comb, sigma_comb), 'k--', label='Normal PDF')

    axs[0].set_title('Combined Portfolio Returns')
    axs[0].set_xlabel('Daily Return')
    axs[0].set_ylabel('Density')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].text(0.02, 0.95,
                f'Skew: {skew_comb:.2f}\nKurtosis: {kurt_comb:.2f}',
                transform=axs[0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # External portfolio
    mu_ext = np.mean(external_daily_returns)
    sigma_ext = np.std(external_daily_returns)
    skew_ext = skew(external_daily_returns)
    kurt_ext = kurtosis(external_daily_returns)

    axs[1].hist(external_daily_returns, bins=50, alpha=0.7, color='green', density=True)

    x_ext = np.linspace(mu_ext - 4*sigma_ext, mu_ext + 4*sigma_ext, 500)
    axs[1].plot(x_ext, norm.pdf(x_ext, mu_ext, sigma_ext), 'k--', label='Normal PDF')

    axs[1].set_title('External-only Portfolio Returns')
    axs[1].set_xlabel('Daily Return')
    axs[1].grid(True)
    axs[1].legend()
    axs[1].text(0.02, 0.95,
                f'Skew: {skew_ext:.2f}\nKurtosis: {kurt_ext:.2f}',
                transform=axs[1].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()

# ------------------ Portfolioweights histogram ------------------
def weights_plot(action_logs, external_trader, test_data, date): 
    combined_list = []
    ext_list = []
    
    num_agents = len(action_logs[0])
    stocks_per_agent = action_logs[0][0].shape[0] - 1  # last col is cash
    total_stocks = num_agents * stocks_per_agent

    for t, step in enumerate(action_logs[0]):
        # step: (num_agents, stocks_per_agent + 1)
        step = np.array(step, dtype=np.float32)
        stock_weights = step[:, :-1].flatten()  # shape (S,)
        cash_weights  = step[:, -1]             # shape (A,)
        total_cash = cash_weights.sum()
        
        agent_w = np.concatenate([stock_weights, [total_cash]])  # shape (S+1,)

        # External weights
        if date in external_trader.index:
            ext_w = external_trader.loc[date].values.astype(np.float32)
            if ext_w.shape[0] == total_stocks:
                # Add cash = 1.0 to external to match agent_w shape
                ext_w = np.concatenate([ext_w, [1.0]])  # or 0.0 if no external cash
        else:
            ext_w = np.zeros_like(agent_w)

        ext_list.append(ext_w)

        # Combine and normalize
        combo = agent_w + ext_w
        combo /= combo.sum()

        combined_list.append(combo)

    # Stack into arrays of shape (T, S+1)
    combined = np.vstack(combined_list)
    ext_only = np.vstack(ext_list)

    # Time-average
    avg_combined = combined.mean(axis=0)
    avg_ext      = ext_only.mean(axis=0)

    # Tickers + cash
    tickers = test_data.columns.tolist() + ['CASH']

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.bar(tickers, avg_combined)
    ax1.set_ylabel("Average Combined Weight")
    ax1.set_title("Combined Portfolio (Including Cash)")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax2.bar(tickers, avg_ext)
    ax2.set_ylabel("Average External Weight")
    ax2.set_xlabel("Asset (Stocks + Cash)")
    ax2.tick_params(axis="x", rotation=90, labelsize=6)

    plt.tight_layout()
    plt.show()


# ------------------ Stock returns over the test period ------------------
def market_returns(test_data):

    # Normalize all stock prices to start at 100
    normalized_prices = test_data / test_data.iloc[0] * 100

    # Calculate daily returns
    daily_returns = normalized_prices.pct_change().dropna()

    # Calculate rolling 20-day Sharpe ratio
    mean_rolling = daily_returns.mean(axis=1).rolling(window=20).mean()
    std_rolling = daily_returns.mean(axis=1).rolling(window=20).std()
    sharpe_rolling = mean_rolling / (std_rolling + 1e-8)  # avoid division by zero

    # Plot rolling Sharpe ratio
    plt.figure(figsize=(12, 6))
    plt.plot(sharpe_rolling.index, sharpe_rolling, label='Rolling 20-Day Sharpe Ratio', color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    plt.title("Rolling 20-Day Sharpe Ratio of Average Market Returns")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend(loc="upper left", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Learning Curve ------------------
def learning_curve(sharpe_per_episode, title='', xlabel='Episodes', ylabel='Sharpe Ratio'):
    plt.figure(figsize=(10, 5))
    
    episodes = np.arange(len(sharpe_per_episode))
    
    # Scatter plot
    plt.scatter(episodes, sharpe_per_episode, label='Sharpe Ratio', color='blue', alpha=0.6)

    # Line of best fit
    coeffs = np.polyfit(episodes, sharpe_per_episode, deg=1)
    trendline = np.polyval(coeffs, episodes)
    plt.plot(episodes, trendline, color='green', label='Trend Line (Best Fit)', linewidth=2)

    # Mean line
    plt.axhline(y=np.mean(sharpe_per_episode), color='red', linestyle='--', label='Mean Sharpe Ratio')

    # Labels and styling
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_training_diagnostics(logs, agent_id=0):
    """
    Plots training diagnostics for a given agent.

    Parameters
    ----------
    logs : dict
        {
            "avg_entropy": List[float],
            "critic_loss": List[float],
            "actor_loss": List[float],
            "grad_norms": Dict[str, List[float]]
        }
    agent_id : int
        Identifier for labeling plots.
    """
    episodes = np.arange(len(logs["avg_entropy"]))

    # ------------------- Entropy Plot -------------------
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, logs["avg_entropy"], label="Avg Entropy", color="blue")
    plt.title(f"[Agent {agent_id}] Avg Entropy Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- Critic Loss Plot -------------------
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, logs["critic_loss"], label="Critic Loss", color="red")
    plt.title(f"[Agent {agent_id}] Critic MSE Loss Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- Actor Loss Plot -------------------
    plt.figure(figsize=(10, 4))
    plt.plot(episodes, logs["actor_loss"], label="Actor Loss", color="green")
    plt.title(f"[Agent {agent_id}] Actor Loss Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------- Gradient Norms Plot -------------------
    for layer, values in logs["grad_norms"].items():
        plt.figure(figsize=(10, 4))
        plt.plot(episodes, values, label=f"{layer} Grad Norm")
        plt.title(f"[Agent {agent_id}] Gradient Norm for {layer}")
        plt.xlabel("Episode")
        plt.ylabel("Grad Norm")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


