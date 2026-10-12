import yfinance as yf
import pandas as pd

def fetch_data(symbol, n_bars=1000, period="25y", interval="1d"):
    """
    Fetch OHLCV data for given symbol using yFinance.
    Parameters:
        symbol   : stock ticker (e.g., 'RELIANCE.NS')
        n_bars   : optional, used if period=None
        period   : e.g. "1y", "5y", "25y"
        interval : "1d", "1h", etc.
    """
    try:
        print(f"[Fetching] {symbol} | period={period}, interval={interval}")
        df = yf.download(symbol + ".NS", period=period, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume"
        })
        return df.dropna()

    except Exception as e:
        print(f"[Error fetching {symbol}]: {e}")
        return pd.DataFrame()
