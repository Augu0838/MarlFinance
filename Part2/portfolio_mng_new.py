import numpy as np
import pandas as pd


def external_weights_new(
    df: pd.DataFrame,
    momentum_lookback: int = 50,
    vol_lookback: int = 20,
    meanrev_short: int = 5,
    meanrev_long: int = 20,
    top_quantile: float = 0.1,
    band: float = 1.0
) -> pd.DataFrame:
    """
    Vectorized combination of three strategy weights:
    - Short-term momentum (long only)
    - Volatility breakout (long only)
    - Mean-reversion (long only)

    Returns a DataFrame of daily weights (rows aligned to df.index[start:]).
    """
    # Determine the first valid date
    start = max(vol_lookback, meanrev_long)
    dates = df.index
    prices = df

    # 1) Momentum: pct change over lookback
    mom = prices.pct_change(periods=momentum_lookback).iloc[start:].values  # (T', S)
    # Threshold for top quantile
    mom_thr = np.quantile(mom, 1 - top_quantile, axis=1, keepdims=True)
    mom_mask = (mom >= mom_thr).astype(np.float32)
    mom_w = mom_mask / np.maximum(mom_mask.sum(axis=1, keepdims=True), 1)

    # 2) Volatility breakout: (P_t_minus1 - mu) / sigma
    mu = prices.rolling(vol_lookback).mean().shift(1).iloc[start:].values
    sigma = prices.rolling(vol_lookback).std().shift(1).add(1e-6).iloc[start:].values
    price_t = prices.shift(1).iloc[start:].values
    b = (price_t - mu) / sigma
    vol_thr = np.quantile(b, 1 - top_quantile, axis=1, keepdims=True)
    vol_mask = (b > np.maximum(vol_thr, band)).astype(np.float32)
    vol_w = vol_mask / np.maximum(vol_mask.sum(axis=1, keepdims=True), 1)

    # 3) Mean-reversion: (short_mean - long_mean) / long_std
    short_mean = prices.rolling(meanrev_short).mean().shift(1).iloc[start:].values
    long_mean = prices.rolling(meanrev_long).mean().shift(1).iloc[start:].values
    long_std = prices.rolling(meanrev_long).std().shift(1).add(1e-6).iloc[start:].values
    z = (short_mean - long_mean) / long_std
    mr_thr_low = np.quantile(z, top_quantile, axis=1, keepdims=True)
    mr_mask = (z <= mr_thr_low).astype(np.float32)  # most oversold
    mr_w = mr_mask / np.maximum(mr_mask.sum(axis=1, keepdims=True), 1)

    # Combine all weights and renormalize
    combined = mom_w + vol_w + mr_w
    combined = combined / np.maximum(combined.sum(axis=1, keepdims=True), 1)

    # Build output DataFrame
    return pd.DataFrame(
        combined,
        index=dates[start:],
        columns=df.columns
    )