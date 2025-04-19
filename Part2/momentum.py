#%%
# ------------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from func import download_close_prices          # already in your repo

#%%
# ------------------------------------------------------------------
# 0.  Define traders
# ------------------------------------------------------------------
class MomentumTrader:
    """Short‑term momentum portfolio constructor."""

    def __init__(self,
                 num_stocks: int,
                 lookback_short: int = 5,
                 lookback_long: int = 20,
                 top_quantile: float = 0.10,
                 dollar_neutral: bool = True):
        self.num_stocks = num_stocks
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.top_quantile = top_quantile
        self.dollar_neutral = dollar_neutral
        self.current_weights = np.zeros(num_stocks, dtype=np.float32)

    def _calc_momentum_score(self, window: pd.DataFrame) -> np.ndarray:
        """Momentum score = return over lookback_short period."""
        prices = window.values

        if prices.shape[0] < self.lookback_short:
            raise ValueError("Window too short for configured lookback")

        short_slice = prices[-self.lookback_short:]
        momentum = (short_slice[-1] - short_slice[0]) / short_slice[0]
        return momentum  # vector of stock momentum scores

    def generate_weights(self, window: pd.DataFrame) -> np.ndarray:
        """Long top performers only."""
        momentum = self._calc_momentum_score(window)

        k = max(1, int(self.num_stocks * self.top_quantile))
        long_idx = np.argsort(momentum)[-k:]  # top momentum

        weights = np.zeros(self.num_stocks, dtype=np.float32)
        long_w = 1.0 / len(long_idx)
        weights[long_idx] = long_w

        self.current_weights = weights
        return weights

    def step(self, window: pd.DataFrame) -> np.ndarray:
        return self.generate_weights(window)

#%%
# ------------------------------------------------------------------
# 1.  Load data
# ------------------------------------------------------------------
num_stocks = 50

tickers = pd.read_csv(
    "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
).iloc[:num_stocks, 0].tolist()

data = download_close_prices(
    tickers, start_day="2022-01-01", period_days=365 * 2
).dropna()

print(data.head())
#%%
# ------------------------------------------------------------------
# 2.  Set up the external trader
# ------------------------------------------------------------------
trader = MomentumTrader(
    num_stocks      = num_stocks,
    lookback_short  = 5,
    lookback_long   = 50,
    top_quantile    = 0.10,
    dollar_neutral  = False        # set False for long‑only
)

window_size = trader.lookback_long          # 20 in this example
weight_log  = []                            # will hold one vector per day
date_index  = []

#%%
# ------------------------------------------------------------------
# 3.  Slide through time, one day at a time
# ------------------------------------------------------------------
for t in range(window_size, len(data)):
    window = data.iloc[t - window_size : t]  # DataFrame (20 × 50)
    w      = trader.step(window)             # numpy array len=50

    weight_log.append(w)
    date_index.append(data.index[t])         # remember the date

weights_df = pd.DataFrame(weight_log, index=date_index, columns=tickers)
print(weights_df.tail())

# %%
# ------------------------------------------------------------------
# 4A.  Rolling‑10‑day Sharpe ratio of the strategy
# ------------------------------------------------------------------
stock_returns = data.pct_change().loc[weights_df.index]         # align dates
portfolio_returns = (weights_df * stock_returns).sum(axis=1)    # daily PnL

win = 10                                                        # 10‑day window
rolling_sharpe = (
    portfolio_returns.rolling(win).mean() /
    (portfolio_returns.rolling(win).std() + 1e-9)               # avoid div‑0
)

plt.figure(figsize=(12, 4))
rolling_sharpe.plot()
plt.title(f"Rolling {win}-Day Sharpe Ratio (momentum trader)")
plt.ylabel("Sharpe ratio")
plt.grid(alpha=0.3)
plt.show()

# %%
# ------------------------------------------------------------------
# 4B.  Bar‑chart of *non‑zero* weights on the last trading day
#       – stock symbols on the x‑axis
# ------------------------------------------------------------------
last_weights = weights_df.iloc[-1]                 # most recent row
nonzero_last = last_weights[last_weights != 0].sort_values()

plt.figure(figsize=(max(6, 0.4 * len(nonzero_last)), 4))
colors = ["tab:green" if w > 0 else "tab:red" for w in nonzero_last]
plt.bar(nonzero_last.index, nonzero_last.values, color=colors)

plt.title("Portfolio Weights on the Last Trading Day")
plt.ylabel("Weight")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
# %%
