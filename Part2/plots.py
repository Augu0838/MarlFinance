import numpy as np
import matplotlib.pylab as plt
from scipy.stats import skew, kurtosis

# ------------------ 1. Cumulative Returns ------------------
def cumulative_returns(dates, combined_daily_returns, external_daily_returns):

    # Number of initial days to skip and rebased to 1 at that point
    offset=100

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

# ------------------ 2. Histograms with Stats ------------------
def histogram(combined_daily_returns, external_daily_returns):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Combined portfolio
    skew_combined = skew(combined_daily_returns)
    kurt_combined = kurtosis(combined_daily_returns)

    axs[0].hist(combined_daily_returns, bins=50, alpha=0.7, color='blue', density=True)
    axs[0].set_title('Combined Portfolio Returns')
    axs[0].set_xlabel('Daily Return')
    axs[0].set_ylabel('Density')
    axs[0].grid(True)
    axs[0].text(0.02, 0.95,
                f'Skew: {skew_combined:.2f}\nKurtosis: {kurt_combined:.2f}',
                transform=axs[0].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # External portfolio
    skew_external = skew(external_daily_returns)
    kurt_external = kurtosis(external_daily_returns)

    axs[1].hist(external_daily_returns, bins=50, alpha=0.7, color='green', density=True)
    axs[1].set_title('External-only Portfolio Returns')
    axs[1].set_xlabel('Daily Return')
    axs[1].grid(True)
    axs[1].text(0.02, 0.95,
                f'Skew: {skew_external:.2f}\nKurtosis: {kurt_external:.2f}',
                transform=axs[1].transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.show()