import os

# ---------------------------------
# Data Config
# ---------------------------------
USE_TV = False  # agar tvdatafeed use karna ho to True karo
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"

# ---------------------------------
# Risk Config
# ---------------------------------
RISK = {
    "fraction": 0.02,   # 2% per trade
    "stop_pct": 0.02,   # 2% stop-loss
    "target_pct": 0.04  # 4% target
}

# ---------------------------------
# Backtest Config
# ---------------------------------
DEFAULT_CAPITAL = 100000  # starting capital
DEFAULT_COMMISSION = 0.001  # 0.1% commission

# ---------------------------------
# Results Path
# ---------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)
