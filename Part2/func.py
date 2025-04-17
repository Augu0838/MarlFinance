# Helper functions

import yfinance as yf
from datetime import datetime, timedelta

def download_close_prices(tickers, start_day, period_days):
    """
    Download close prices for `tickers` from `start_day` for `period_days` days.
    
    Parameters
    ----------
    tickers : list or str
        One or more ticker symbols (e.g. ['AAPL','MSFT'] or 'AAPL').
    start_day : str
        Start date in 'YYYY-MM-DD' format.
    period_days : int
        Number of calendar days to include after start_day.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame of closing prices indexed by date.
    """
    # parse the start date
    start = datetime.strptime(start_day, "%Y-%m-%d")
    # compute the end date by adding the desired timedelta
    end   = start + timedelta(days=period_days)
    
    # format back to string for yfinance
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")
    
    # download and return only the 'Close' columns
    data = yf.download(tickers, start=start_str, end=end_str)["Close"]
    return data
