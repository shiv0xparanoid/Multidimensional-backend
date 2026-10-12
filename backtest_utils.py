# app/backtest_utils.py
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from backtesting import Backtest
from app.strategies import (
    MovingAverageStrategy, MomentumStrategy,
    MeanReversionStrategy, RSIStrategy, EnsembleStrategy
)
from app.config import DEFAULT_CAPITAL, DEFAULT_COMMISSION, RISK

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


STRATEGIES = {
    "ma": MovingAverageStrategy,
    "momentum": MomentumStrategy,
    "meanrev": MeanReversionStrategy,
    "rsi": RSIStrategy,
    "ensemble": EnsembleStrategy
}


def run_backtest_for_symbol(df, sym, cash=DEFAULT_CAPITAL, commission=DEFAULT_COMMISSION):
    results = []
    for name, strat in STRATEGIES.items():
        try:
            df_bt = df[['open','high','low','close','volume']].copy()
            df_bt.columns = ['Open','High','Low','Close','Volume']  # backtesting requirement

            bt = Backtest(df_bt, strat, cash=cash, commission=commission, exclusive_orders=True, finalize_trades=True)
            stats = bt.run()
            # save html interactive plot
            plot_path = os.path.join(RESULTS_DIR, f"{sym}_{name}.html")
            try:
                bt.plot(open_browser=False, filename=plot_path)
            except Exception as e:
                print("[Plot error]", e)

            # attach symbol/strategy
            stats["symbol"] = sym
            stats["strategy"] = name

            # check max drawdown cutoff (flag)
            mdd = stats.get("Max. Drawdown [%]", None)
            if mdd is not None and abs(mdd) > RISK.get("max_drawdown_cutoff_pct", 0.25) * 100:
                stats["dd_cutoff_flag"] = True
            else:
                stats["dd_cutoff_flag"] = False

            results.append(stats)
        except Exception as e:
            print(f"Error {sym}-{name}: {e}")
    return results


def batch_backtest(tickers, fetch_func, outdir=RESULTS_DIR, n_bars=1500):
    os.makedirs(outdir, exist_ok=True)
    all_results = []
    for sym in tickers:
        print(f"\n=== {sym} ===")
        df = fetch_func(sym, n_bars=n_bars)
        if df is None or df.empty or len(df) < 100:
            print(f"Insufficient data for {sym}")
            continue
        res = run_backtest_for_symbol(df, sym)
        all_results.extend(res)

    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(os.path.join(outdir, "summary.csv"), index=False)
        return df_res
    return pd.DataFrame()
