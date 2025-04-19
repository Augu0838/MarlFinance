# ------------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from func import download_close_prices     

# ------------------------------------------------------------------
# Volatility‑Breakout Trader
# ------------------------------------------------------------------
class VolatilityBreakoutTrader:
    """
    Simple break‑out rule:
        • Compute long_window mean (μ) and std (σ) of each stock.
        • Look at today's close P_t.
        • If P_t > μ + band*σ   ⇒ long (bullish breakout)
        • If P_t < μ - band*σ   ⇒ short (bearish breakout)
        • Otherwise              weight = 0
    The top_quantile strongest breakouts are traded with equal weight.
    """

    def __init__(self,
                 num_stocks: int,
                 lookback_long: int = 20,
                 band: float = 1.0,
                 top_quantile: float = 0.10,
                 dollar_neutral: bool = False):
        """
        long_window   : look‑back length for μ and σ
        band          : width of the breakout band in σ units
        top_quantile  : fraction of stocks to long OR short
        dollar_neutral: if True → +0.5 long and –0.5 short
                        if False → +1.0 long,  0   short  (long‑only)
        """
        self.num_stocks     = num_stocks
        self.lookback_long  = lookback_long
        self.band           = band
        self.top_quantile   = top_quantile
        self.dollar_neutral = dollar_neutral
        self.current_weights = np.zeros(num_stocks, dtype=np.float32)

    # --------------------------- helper ---------------------------
    def _breakout_score(self, window: pd.DataFrame) -> np.ndarray:
        """
        Return a z‑score style breakout measure for each stock:
            b_i = (P_t - μ_i) / σ_i
        Positive  → bullish breakout, negative → bearish.
        """
        long_slice = window.values  # shape (T, S) where T == long_window
        price_t    = long_slice[-1]                          # today's close
        mu         = long_slice.mean(axis=0)
        sigma      = long_slice.std(axis=0) + 1e-6
        return (price_t - mu) / sigma

    # ----------------------- main public API ----------------------
    def generate_weights(self, window: pd.DataFrame) -> np.ndarray:
        """
        Produce a length‑S weight vector.
        """
        b = self._breakout_score(window)

        # pick strongest breakouts
        k = max(1, int(self.num_stocks * self.top_quantile))
        long_idx  = np.argsort(b)[-k:]        # largest positive b
        short_idx = np.argsort(b)[:k]         # most negative b

        weights = np.zeros(self.num_stocks, dtype=np.float32)

        if self.dollar_neutral:
            w_long  =  0.5 / len(long_idx)
            w_short = -0.5 / len(short_idx)
        else:
            w_long  =  1.0 / len(long_idx)
            w_short =  0.0                     # no shorts in long‑only mode

        # apply band filter: only trade if breakout exceeds ±band
        for i in long_idx:
            if b[i] > self.band:
                weights[i] = w_long
        for i in short_idx:
            if b[i] < -self.band:
                weights[i] = w_short

        self.current_weights = weights
        return weights

    # alias so it matches the mean‑reversion interface
    def step(self, window: pd.DataFrame) -> np.ndarray:
        return self.generate_weights(window)

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

# ------------------------------------------------------------------
# 2.  Set up the external trader
# ------------------------------------------------------------------
trader = VolatilityBreakoutTrader(num_stocks=num_stocks)

window_size = trader.lookback_long          # 20 in this example
weight_log  = []                            # will hold one vector per day
date_index  = []

print("Trader defined")

# ------------------------------------------------------------------
# 3.  Slide through time, one day at a time
# ------------------------------------------------------------------
for t in range(window_size, len(data)):
    window = data.iloc[t - window_size : t]  # DataFrame (20 × 50)
    w      = trader.step(window)             # numpy array len=50

    weight_log.append(w)
    date_index.append(data.index[t])         # remember the date

weights_df = pd.DataFrame(weight_log, index=date_index, columns=tickers)
print("Weights found")

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

