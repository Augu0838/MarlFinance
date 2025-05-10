import yfinance as yf
import pandas as pd
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
