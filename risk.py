# app/risk.py
import pandas as pd
import numpy as np

# -------------------------------
# Risk & Position Sizing Utilities
# -------------------------------

def fixed_fraction_size(equity, risk_fraction=0.02, price=100):
    """
    Fixed fraction position sizing.
    Example: 2% of equity risk per trade.
    """
    dollar_alloc = equity * risk_fraction
    size = dollar_alloc / max(price, 1)
    return max(1, int(size))


def atr_position_size(equity, atr, risk=0.01):
    """
    Position sizing using ATR (volatility based).
    Uses risk fraction of equity divided by ATR.
    """
    risk_amount = equity * risk
    size = risk_amount / max(atr, 1e-6)
    return max(1, int(size))


def pct_stop_levels(price, stop_pct=0.02, target_pct=0.04):
    """
    Generate stop-loss and target levels based on % of entry price.
    """
    stop = price * (1 - stop_pct)
    target = price * (1 + target_pct)
    return stop, target


def kelly_fraction(win_rate, avg_win, avg_loss):
    """
    Kelly criterion sizing fraction.
    """
    if avg_loss == 0:
        return 0.01
    b = avg_win / abs(avg_loss)
    kelly = win_rate - (1 - win_rate) / b
    return float(max(0, min(kelly, 1)))


# -------------------------------
# Confidence-based Trading Simulator
# -------------------------------

def simulate_confidence_trading(
    df,
    models,
    threshold=0.7,
    fraction=None,
    commission=0.001,
    initial_equity=100000
):
    """
    Confidence-based trading simulator:
    - Sirf wahi trades leta hai jaha confidence > threshold
    - Confidence ke saath position size scale hota hai
    """
    equity = initial_equity
    equity_curve = []
    trades = []
    n_wins, n_losses = 0, 0

    close = df['Close'].values
    n = len(close)

    for i in range(100, n):  # warmup period
        X = np.array([
            close[i-10:i].mean(),
            close[i-20:i].mean(),
            close[i-50:i].mean(),
            close[i] - close[i-10:i].mean()
        ]).reshape(1, -1)

        probs = []
        for m in models:
            try:
                p = m.predict_proba(X)[0]
                probs.append(p)
            except Exception:
                pass
        if not probs:
            equity_curve.append(equity)
            continue

        avg_prob = np.mean(probs, axis=0)
        conf = max(avg_prob)
        pred = np.argmax(avg_prob)

        if conf < threshold:
            equity_curve.append(equity)
            continue

        if fraction is None:
            frac = min(1.0, (conf - threshold) / (1 - threshold))
        else:
            frac = fraction

        entry = close[i]
        exit_price = close[i+1] if i+1 < n else entry
        pnl = 0

        if pred == 1:  # BUY
            pnl = (exit_price - entry) / entry
        elif pred == 0:  # SELL
            pnl = (entry - exit_price) / entry

        pnl_after_fee = pnl * frac - commission
        equity *= (1 + pnl_after_fee)

        if pnl_after_fee > 0:
            n_wins += 1
        else:
            n_losses += 1

        trades.append(pnl_after_fee)
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve)

    total_return = (equity / initial_equity - 1) * 100
    sharpe = np.mean(trades) / (np.std(trades) + 1e-8) * np.sqrt(252) if trades else 0
    max_dd = (
        (equity_series.cummax() - equity_series).max() /
        equity_series.cummax().max() * 100 if not equity_series.empty else 0
    )

    return {
        "start_equity": initial_equity,
        "end_equity": equity,
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "n_trades": len(trades),
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": n_wins / max(1, (n_wins + n_losses)),
        "equity_series": equity_series
    }
