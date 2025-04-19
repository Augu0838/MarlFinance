#%%
# ------------------------------------------------------------------
# 0.  Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from func import download_close_prices 

def external_weights(num_stocks:int, start_day):
    # ------------------------------------------------------------------
    # 0.  Define traders
    # ------------------------------------------------------------------
    ############ Momentum trader ############
    class MomentumTrader:
        """Short‑term momentum portfolio constructor."""

        def __init__(self,
                    num_stocks: int,
                    lookback_short: int = 50,
                    lookback_long: int = 100,
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

    ############ Volatility breakout trader ############
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
        
    ############ Mean reversion trader ############
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
                    dollar_neutral: bool = False):
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
                short_w = 0.0 / len(short_idx)

            weights[long_idx]  = long_w
            weights[short_idx] = short_w

            # cache and return
            self.current_weights = weights
            return weights

        # ------------------------------------------------------------------
        def step(self, window: pd.DataFrame) -> np.ndarray:
            """Alias so the class mirrors an *agent* API (state → action)."""
            return self.generate_weights(window)

    # ------------------------------------------------------------------
    # 1.  Get data
    # ------------------------------------------------------------------
    num_stocks = num_stocks
    tickers = pd.read_csv(
        "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
    ).iloc[:num_stocks, 0].tolist()

    data = download_close_prices(
        tickers, start_day=start_day, period_days=365 * 3
    ).dropna()


    # ------------------------------------------------------------------
    # 2.  Define traders
    # ------------------------------------------------------------------
    meanRvs_trader = MeanReversionTrader(num_stocks=num_stocks)
    meanRvs_window_size = meanRvs_trader.lookback_long          
    meanRvs_weight_log  = []                            
    meanRvs_date_index  = []

    volBrk_trader = VolatilityBreakoutTrader(num_stocks=num_stocks)
    volBrk_window_size = volBrk_trader.lookback_long         
    volBrk_weight_log  = []                              
    volBrk_date_index  = []

    mmt_trader = MomentumTrader(num_stocks=num_stocks)
    mmt_window_size = mmt_trader.lookback_long          
    mmt_weight_log  = []                                   
    mmt_date_index  = []

    # ------------------------------------------------------------------
    # 3.  Slide through time, one day at a time
    # ------------------------------------------------------------------

    ############ Mean reversion trader ############
    for t in range(meanRvs_window_size, len(data)):
        meanRvs_window = data.iloc[t - meanRvs_window_size : t]     # DataFrame (20 × 50)
        meanRvs_w      = meanRvs_trader.step(meanRvs_window)             # numpy array len=50

        meanRvs_weight_log.append(meanRvs_w)
        meanRvs_date_index.append(data.index[t])                    # remember the date

    meanRvs_weights = pd.DataFrame(meanRvs_weight_log, index=meanRvs_date_index, columns=tickers)

    ############ Volatility breakout trader ############
    for t in range(volBrk_window_size, len(data)):  
        volBrk_window = data.iloc[t - volBrk_window_size : t]       # DataFrame (20 × 50)
        volBrk_w      = volBrk_trader.step(volBrk_window)                # numpy array len=50

        volBrk_weight_log.append(volBrk_w)
        volBrk_date_index.append(data.index[t])                     # remember the date

    volBrk_weights = pd.DataFrame(volBrk_weight_log, index=volBrk_date_index, columns=tickers)

    ############ Momentum trader ############
    for t in range(mmt_window_size, len(data)):
        mmt_window = data.iloc[t - mmt_window_size : t]             # DataFrame (20 × 50)
        mmt_w      = mmt_trader.step(mmt_window)                         # numpy array len=50

        mmt_weight_log.append(mmt_w)
        mmt_date_index.append(data.index[t])                        # remember the date

    mmt_weights = pd.DataFrame(mmt_weight_log, index=mmt_date_index, columns=tickers)

    # ------------------------------------------------------------------
    # 4.  Combine and normalize weights across all strategies
    # ------------------------------------------------------------------

    tables = [meanRvs_weights, volBrk_weights, mmt_weights]

    common_index = tables[0].index
    for tbl in tables[1:]:
        common_index = common_index.intersection(tbl.index)

    aligned = [tbl.loc[common_index] for tbl in tables]   # Align the tables, same order, same dates

    # Stack them, sum across the 1st level, and renormalise
    # shape of each  : (T, 50)
    # after np.sum   : (T, 50)
    combined_raw = sum(aligned)

    # row‑wise L1 normalisation so each date’s weights add to 1
    combined_weights = combined_raw.div(combined_raw.sum(axis=1), axis=0)
    
    return combined_weights

