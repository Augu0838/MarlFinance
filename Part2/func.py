import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def download_close_prices(tickers, start_day, period_days) -> pd.DataFrame:
    """
    One‑shot download of all tickers, then discard the bad ones.

    • No per‑ticker try/except loop.
    • Anything with < min_valid_fraction non‑NaNs is dropped.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    start = datetime.strptime(start_day, "%Y-%m-%d")
    end   = start + timedelta(days=period_days)
    df    = yf.download(
                tickers, start=start, end=end, progress=False
            )["Close"]                         # DataFrame, columns = tickers

    if df.ndim == 1:                           # happens if len(tickers)==1
        df = df.to_frame()

    # keep only those columns that have enough usable data
    valid_fractions = df.notna().mean()
    good_cols = valid_fractions >= 1

    # Print discarded tickers
    discarded = good_cols[~good_cols].index.tolist()
    if discarded:
        print("Discarded tickers due to insufficient data:", discarded)

    clean_df  = df.loc[:, good_cols].dropna(how="all")

    if clean_df.empty:
        raise ValueError("No tickers with sufficient data were downloaded.")

    print(f"Number of stocks returned: {clean_df.shape[1]}")
    return clean_df


def rolling_sharpe(returns, window):
    returns = pd.Series(returns)
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std() + 1e-6
    return (mean / std).values

def process_results(df, test_data, action_logs, external_trader,window_size):

    # Extract necessary data
    returns = np.diff(test_data.values, axis=0) / test_data.values[:-1]  # shape (T-1, S)
    dates = test_data.index[1:]  # align with returns

    # Determine actual usable length (safe length after start_idx)
    eval_len = len(action_logs[0])  # number of timesteps in the episode
    start_idx = window_size
    max_len = min(eval_len, len(dates) - start_idx)

    # Align dates and return windows safely
    eval_dates = dates[start_idx : start_idx + max_len]
    ret_window = returns[start_idx : start_idx + max_len]

    combined_daily_returns = []

    for t in range(max_len):
        step_actions = action_logs[0][t]
        agent_weights = np.vstack(step_actions).astype(np.float32)
        stock_weights = agent_weights[:, :-1]
        cash_weights  = agent_weights[:, -1]
        combined_stock = stock_weights.flatten()
        combined_cash  = np.sum(cash_weights)
        final_agent_portfolio = np.concatenate([combined_stock, [combined_cash]])
        final_agent_portfolio /= final_agent_portfolio.sum()
        date = eval_dates[t]
    
        if date in external_trader.index:
            ext_weights = external_trader.loc[date].values.astype(np.float32)

            # Ensure shapes match
            if ext_weights.shape[0] != final_agent_portfolio.shape[0]:
                raise ValueError(f"Shape mismatch at {date}: ext={ext_weights.shape[0]}, agent={final_agent_portfolio.shape[0]}")

            alpha = 0.5
            combo_weights = alpha * final_agent_portfolio + (1 - alpha) * ext_weights
        else:
            combo_weights = final_agent_portfolio

        r = np.dot(ret_window[t], combo_weights[:-1])  # Exclude cash weight
        combined_daily_returns.append(r)

    external_daily_returns = []
    for t in range(max_len):
        date = eval_dates[t]
        if date in external_trader.index:
            weights = external_trader.loc[date].values
            r = np.dot(ret_window[t], weights[:-1])
            external_daily_returns.append(r)
        else:
            external_daily_returns.append(0.0)  # fallback if date not available

    sharpe_combined = np.nan_to_num(rolling_sharpe(combined_daily_returns, window_size), nan=0.0)
    sharpe_external = np.nan_to_num(rolling_sharpe(external_daily_returns, window_size), nan=0.0)

    return_df = pd.DataFrame({
        'Sharpe Combined': sharpe_combined,
        'Sharpe External': sharpe_external,
        'External Daily Returns': external_daily_returns,
        'Combined Daily Returns': combined_daily_returns,
    }, index=eval_dates)

    return return_df
