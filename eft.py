import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
start_date = "2010-01-01"
end_date = "2025-01-01"
trading_days = 252

# -------------------------------
# 1. Download 30-Year US Treasury Yields
# -------------------------------
tyx_data = yf.download("^TYX", start=start_date, end=end_date)

treasury_yield = tyx_data['Close']

# Convert the yield (in %) to a daily rate (e.g., 2.50% becomes 0.025/252)
daily_rfr = (treasury_yield / 100) / trading_days
daily_rfr.name = "daily_rfr"

# -------------------------------
# 2. Download ETF Data
# -------------------------------
etfs = {
    'S&P500': 'SPY',
    'ClimateInvestment': 'ICLN',  # iShares Global Clean Energy ETF
    'Manufacturing': 'XLI'         # Industrial Select Sector SPDR Fund
}

prices = pd.DataFrame()
for name, ticker in etfs.items():
    data = yf.download(ticker, start=start_date, end=end_date)
    prices[name] = data['Close']

# -------------------------------
# 3. Compute Daily Returns for ETFs
# -------------------------------
returns = prices.pct_change().dropna()

# -------------------------------
# 4. Align and Assign the Daily Risk-Free Rate
# -------------------------------
# Reindex the daily_rfr series to match the returns index and forward fill any missing values
daily_rfr_aligned = daily_rfr.reindex(returns.index, method='ffill')

# Directly assign the aligned series to a new column in returns
returns['daily_rfr'] = daily_rfr_aligned

# Confirm that the column is there
print("Returns with risk-free rate column:")
print(returns.head())

# Continue with your Sharpe ratio calculations...
# Example: Compute Sharpe Ratio using the dynamic risk-free rate
sharpe_dynamic = {}
for col in etfs.keys():
    # Calculate excess returns: ETF daily return minus the daily risk-free rate
    excess_returns = returns[col] - returns['daily_rfr']
    sharpe_dynamic[col] = (excess_returns.mean() / excess_returns.std()) * np.sqrt(trading_days)

print("Sharpe Ratios (Dynamic Risk-Free Rate):")
print(sharpe_dynamic)
