import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_close_prices(tickers, start_day, period_days,
                          min_valid_fraction: float = 0.95) -> pd.DataFrame:
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
    good_cols = df.notna().mean() >= min_valid_fraction
    clean_df  = df.loc[:, good_cols].dropna(how="all")

    if clean_df.empty:
        raise ValueError("No tickers with sufficient data were downloaded.")

    return clean_df
