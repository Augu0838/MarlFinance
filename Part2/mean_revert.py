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
class MeanReversionTrader:
    """Short‑term mean‑reversion portfolio constructor.

    Parameters
    ----------
    num_stocks : int
        Total number of stocks in the universe (e.g. 500)
    lookback_short : int, default 5
        Number of most‑recent days used to measure the short‑term move.
    lookback_long : int, default 20
        Number of days used to compute the long‑term mean (the anchor).
    top_quantile : float, default 0.10
        Fraction of stocks to long (most oversold) and to short (most overbought).
    dollar_neutral : bool, default True
        If True → portfolio is 50 % long / 50 % short so net weight sums to 0.
        If False → full 100 % net long exposure (long weights sum to +1).
    """

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

        # keep a copy of last weights (handy for turnover or costs)
        self.current_weights = np.zeros(num_stocks, dtype=np.float32)

    # ------------------------------------------------------------------
    def _calc_zscore(self, window: pd.DataFrame) -> np.ndarray:
        """Return a z‑score for every stock in the window.

        z_i = ( short_mean_i − long_mean_i ) / long_std_i
        where the means/std are taken over lookback_short / lookback_long.
        """
        prices = window.values  # shape (T, S)

        if prices.shape[0] < max(self.lookback_long, self.lookback_short):
            raise ValueError("Window too short for configured lookbacks")

        short_slice = prices[-self.lookback_short:]
        long_slice  = prices[-self.lookback_long:]

        short_mean = short_slice.mean(axis=0)
        long_mean  = long_slice.mean(axis=0)
        long_std   = long_slice.std(axis=0) + 1e-6  # avoid div/0

        z = (short_mean - long_mean) / long_std
        return z  # vector length = num_stocks

    # ------------------------------------------------------------------
    def generate_weights(self, window: pd.DataFrame) -> np.ndarray:
        """Return **full‑universe weight vector** (length = num_stocks).

        Long the bottom `top_quantile` of z‑scores, short the top `top_quantile`.
        Weights are equal‑weighted inside long and short buckets.
        """
        z = self._calc_zscore(window)

        # ranks: lowest z → oversold, highest z → overbought
        k = max(1, int(self.num_stocks * self.top_quantile))
        long_idx  = np.argsort(z)[:k]          # most negative
        short_idx = np.argsort(z)[-k:]         # most positive

        weights = np.zeros(self.num_stocks, dtype=np.float32)

        if self.dollar_neutral:
            long_w  =  0.5 / len(long_idx)
            short_w = -0.5 / len(short_idx)
        else:
            long_w  =  1.0 / len(long_idx)
            short_w = -1.0 / len(short_idx)

        weights[long_idx]  = long_w
        weights[short_idx] = short_w

        # cache and return
        self.current_weights = weights
        return weights

    # ------------------------------------------------------------------
    def step(self, window: pd.DataFrame) -> np.ndarray:
        """Alias so the class mirrors an *agent* API (state → action)."""
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
trader = MeanReversionTrader(
    num_stocks      = num_stocks,
    lookback_short  = 5,
    lookback_long   = 50,
    top_quantile    = 0.10,
    dollar_neutral  = True        # set False for long‑only
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
plt.title(f"Rolling {win}-Day Sharpe Ratio (mean‑reversion trader)")
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
