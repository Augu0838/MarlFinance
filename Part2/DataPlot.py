import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def analyze_equal_weight_index(tickers, start, end, base_value=100):
    import yfinance as yf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Download adjusted close prices
    price_df = yf.download(tickers, start=start, end=end, progress=False)['Close']
    price_df = price_df.dropna(axis=1, how='any')

    # Normalize each column to 1 at start
    norm_prices = price_df / price_df.iloc[0]

    # Equal-weighted index
    equal_weight_index = norm_prices.mean(axis=1)
    scaled_index = equal_weight_index * base_value

    # Compute statistics
    returns = scaled_index.pct_change().dropna()
    variance = returns.var()
    volatility = returns.std()
    avg_return = returns.mean()
    sharpe_ratio = avg_return / volatility
    sharpe_annualized = sharpe_ratio * np.sqrt(252)

    high = scaled_index.max()
    high_date = scaled_index.idxmax()
    low = scaled_index.min()
    low_date = scaled_index.idxmin()
    overall_return = (scaled_index.iloc[-1] / scaled_index.iloc[0] - 1) * 100

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(scaled_index)
    plt.title("S&P 500 Equal-Weighted Index (Subset)")
    plt.xlabel("Date")
    plt.ylabel("Index Level")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print stats
    print(f"Portfolio variance: {variance:.6f}")
    print(f"High: {high:.2f} on {high_date.date()}")
    print(f"Low: {low:.2f} on {low_date.date()}")
    print(f"Overall return: {overall_return:.2f}%")
    print(f"Sharpe ratio (daily): {sharpe_ratio:.4f}")
    print(f"Sharpe ratio (annualized): {sharpe_annualized:.4f}")

    return scaled_index


start_day = "2021-01-01"
start = datetime.strptime(start_day, "%Y-%m-%d")
end   = start + timedelta(days=365*4)

num_stocks = 501
tickers = pd.read_csv(
    "https://github.com/Augu0838/MarlFinance/blob/main/Part2/sp500_tickers.csv?raw=true"
).iloc[:num_stocks+1, 0].tolist()

print(start, end)
analyze_equal_weight_index(tickers, start, end)