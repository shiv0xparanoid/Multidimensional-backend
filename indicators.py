# app/indicators.py
import pandas as pd
import numpy as np

def sma(series, n: int):
    """
    Simple Moving Average
    """
    series = pd.Series(series).squeeze()
    if hasattr(series, "values") and series.ndim > 1:
        series = series.values.ravel()
        series = pd.Series(series)
    return series.rolling(window=n).mean()

def rsi(series, n: int = 14):
    """
    Relative Strength Index (RSI)
    """
    series = pd.Series(series).squeeze()
    if hasattr(series, "values") and series.ndim > 1:
        series = series.values.ravel()
        series = pd.Series(series)

    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    gain = up.rolling(window=n).mean()
    loss = down.rolling(window=n).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, n: int = 14):
    """
    Average True Range (ATR)
    df must have columns: high, low, close
    """
    high = df['high'].squeeze()
    low = df['low'].squeeze()
    close = df['close'].squeeze()

    # Flatten if 2D
    if hasattr(high, "values") and high.ndim > 1:
        high = pd.Series(high.values.ravel())
    if hasattr(low, "values") and low.ndim > 1:
        low = pd.Series(low.values.ravel())
    if hasattr(close, "values") and close.ndim > 1:
        close = pd.Series(close.values.ravel())

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(window=n).mean()

def momentum(series, n: int = 10):
    """
    Momentum Indicator: (Price_today / Price_n_days_ago - 1)
    """
    series = pd.Series(series).squeeze()
    if hasattr(series, "values") and series.ndim > 1:
        series = series.values.ravel()
        series = pd.Series(series)
    return series.pct_change(periods=n)
