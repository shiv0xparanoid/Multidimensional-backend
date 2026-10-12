# app/portfolio.py

import pandas as pd
import os

RESULTS_DIR = "results"

def build_portfolio(df_results, capital=1_000_000):
    best_strategies = []
    for sym, group in df_results.groupby("symbol"):
        # pick strategy with max Return [%]
        best = group.loc[group["Return [%]"].idxmax()]
        sharpe = best.get("Sharpe", best.get("Sharpe Ratio", None))  # ✅ Safe access
        best_strategies.append({
            "symbol": sym,
            "strategy": best["strategy"],
            "Return [%]": best["Return [%]"],
            "Sharpe": sharpe
        })

    df_best = pd.DataFrame(best_strategies)
    df_best.to_csv(os.path.join(RESULTS_DIR, "best_strategies.csv"), index=False)

    avg_return = df_best["Return [%]"].mean()
    avg_sharpe = df_best["Sharpe"].mean()
    gross_roi = (df_best["Return [%]"].sum())

    portfolio_stats = {
        "Stocks": len(df_best),
        "Capital": capital,
        "Final Capital": capital * (1 + avg_return / 100),
        "Average Return %": avg_return,
        "Gross Portfolio ROI %": gross_roi,
        "Average Sharpe": avg_sharpe,
    }

    return portfolio_stats, df_best
