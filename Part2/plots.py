import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm, skew, kurtosis

# ------------------ Sharpe by episode ------------------
def plot_training_sharpe(sharpe_series):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(sharpe_series) + 1), sharpe_series, marker='o')
    plt.title("Average Sharpe Ratio per Training Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Sharpe Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Sharpe ratio ------------------
def sharpe_ratios(sharpe_combined, sharpe_external):
    days = np.arange(len(sharpe_combined))

    plt.figure(figsize=(10, 5))
    plt.plot(days, sharpe_combined, label='Combined Portfolio')
    plt.plot(days, sharpe_external, label='External-only Portfolio')
    plt.title('10-Day Rolling Sharpe Ratio')
    plt.xlabel('Day')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------ Sharpe difference ------------------
def sharp_difference(sharpe_combined, sharpe_external):
    # Compute difference between the rolling Sharpe ratios
    sharpe_diff = np.array(sharpe_combined) - np.array(sharpe_external)
    days = np.arange(len(sharpe_combined))

    # Plot the difference
    plt.figure(figsize=(10, 5))
    plt.plot(days, sharpe_diff, label='Sharpe Ratio Difference (Combined - External)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('10-Day Rolling Sharpe Ratio Difference')
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