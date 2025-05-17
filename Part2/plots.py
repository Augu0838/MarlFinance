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
def sharpe_ratios(sharpe_combined, sharpe_external, title='10-Day Rolling Sharpe Ratio'):
    days = np.arange(len(sharpe_combined))

    plt.figure(figsize=(10, 5))
    plt.plot(days, sharpe_combined, label='Combined Portfolio')
    plt.plot(days, sharpe_external, label='External-only Portfolio')
    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
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
    ext_list      = []
    
    for t, step in enumerate(action_logs[0]):
        # step: (num_agents, stocks_per_agent)
        agent_w = step.flatten()   # shape (S,)

        # fetch external weights (or zeros if missing)
        if date in external_trader.index:
            ext_w = external_trader.loc[date].values.astype(np.float32)
        else:
            ext_w = np.zeros_like(agent_w)

        ext_list.append(ext_w)

        # combine and renormalize
        combo = agent_w + ext_w
        combo /= combo.sum()

        combined_list.append(combo)

    # stack into arrays of shape (T, S)
    combined = np.vstack(combined_list)
    ext_only = np.vstack(ext_list)

    # time‐average across the evaluation episode
    avg_combined = combined.mean(axis=0)
    avg_ext      = ext_only.mean(axis=0)

    tickers = test_data.columns.tolist()  # length S

    # create a figure with two rows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # top: combined weights
    ax1.bar(tickers, avg_combined)
    ax1.set_ylabel("Average Combined Weight")
    ax1.set_title("Combined Portfolio vs. External Strategy")
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # bottom: external‐only weights
    ax2.bar(tickers, avg_ext)
    ax2.set_ylabel("Average External Weight")
    ax2.set_xlabel("Stock Ticker")
    ax2.tick_params(axis="x", rotation=90, labelsize=6)

    plt.tight_layout()
    plt.show()

# ------------------ Stock returns over the test period ------------------
def market_returns(test_data):

    # Normalize all stock prices to start at 100
    normalized_prices = test_data / test_data.iloc[0] * 100

    # Calculate daily returns
    daily_returns = normalized_prices.pct_change().dropna()

    # Calculate cumulative returns
    cumulative_returns = (1 + daily_returns).cumprod()

    # compute mean cumulative returns
    mean_cumulative_returns = cumulative_returns.mean(axis=1)
    cumulative_returns['Mean'] = mean_cumulative_returns

    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns['Mean'], label='Average cummulative returns', alpha=0.7)

    plt.title("Cumulative Returns of Stocks During Evaluation Period")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend(loc="upper left", fontsize="small", ncol=2)
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