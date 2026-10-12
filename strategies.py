# app/strategies.py
import numpy as np
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from app.indicators import sma, rsi, atr
from app.risk import (
    fixed_fraction_size, atr_position_size, pct_stop_levels, kelly_fraction
)
from app.config import RISK

# Utility helper inside strategies
def _safe_get_equity(strategy):
    try:
        return float(strategy.equity)
    except Exception:
        try:
            return float(strategy.broker.equity)
        except Exception:
            return 100000.0


class BaseRiskStrategy(Strategy):
    fixed_fraction = RISK.get("fixed_fraction", 0.02)
    atr_mult = RISK.get("atr_mult", 1.5)
    risk_per_trade = RISK.get("risk_per_trade", 0.01)
    stop_loss_pct = RISK.get("stop_loss_pct", 0.03)
    take_profit_pct = RISK.get("take_profit_pct", 0.06)
    kelly_enabled = RISK.get("kelly_enabled", False)

    def init(self):
        self.close = self.data.Close
        self.entry_price = None
        self.stop_price = None
        self.take_price = None

    def size_from_equity(self, price, atr_val):
        equity = _safe_get_equity(self)
        if self.kelly_enabled:
            fraction = self.fixed_fraction
        else:
            fraction = self.fixed_fraction

        if atr_val is not None and atr_val > 0:
            shares = atr_position_size(equity, atr_val * self.atr_mult, risk=self.risk_per_trade)
            if shares > 0:
                return shares

        dollar_alloc = fixed_fraction_size(equity, fraction, price)
        return max(0, int(dollar_alloc))

    def set_stops_on_entry(self, entry_price, atr_val=None):
        if atr_val is not None and atr_val > 0:
            stop = entry_price - atr_val * self.atr_mult
            take = entry_price + atr_val * self.atr_mult * 2
        else:
            stop, take = pct_stop_levels(entry_price, stop_pct=self.stop_loss_pct, target_pct=self.take_profit_pct)
        self.entry_price, self.stop_price, self.take_price = entry_price, stop, take

    def check_stops(self):
        if not self.position:
            return
        price = self.data.Close[-1]
        if self.stop_price is not None and price <= self.stop_price:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None
        elif self.take_price is not None and price >= self.take_price:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None


class MovingAverageStrategy(BaseRiskStrategy):
    short = 20
    long = 50

    def init(self):
        super().init()
        self.ma_short = self.I(sma, self.data.Close, self.short)
        self.ma_long = self.I(sma, self.data.Close, self.long)
        self.atr14 = self.I(atr, pd.DataFrame({
            "high": self.data.High, "low": self.data.Low, "close": self.data.Close
        }), 14)

    def next(self):
        self.check_stops()
        price = self.data.Close[-1]
        atr_val = float(self.atr14[-1]) if len(self.atr14) and not np.isnan(self.atr14[-1]) else None
        if crossover(self.ma_short, self.ma_long):
            shares = self.size_from_equity(price, atr_val)
            if shares > 0 and not self.position.is_long:
                self.buy(size=shares)
                self.set_stops_on_entry(price, atr_val)
        elif crossover(self.ma_long, self.ma_short) and self.position.is_long:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None


class MomentumStrategy(BaseRiskStrategy):
    mom_period = 20

    def init(self):
        super().init()
        self.mom = self.I(lambda x: pd.Series(x).pct_change(self.mom_period).to_numpy(), self.data.Close)
        self.atr14 = self.I(atr, pd.DataFrame({
            "high": self.data.High, "low": self.data.Low, "close": self.data.Close
        }), 14)

    def next(self):
        self.check_stops()
        price = self.data.Close[-1]
        atr_val = float(self.atr14[-1]) if len(self.atr14) and not np.isnan(self.atr14[-1]) else None
        if self.mom[-1] > 0 and not self.position.is_long:
            shares = self.size_from_equity(price, atr_val)
            if shares > 0:
                self.buy(size=shares)
                self.set_stops_on_entry(price, atr_val)
        elif self.mom[-1] <= 0 and self.position.is_long:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None


class MeanReversionStrategy(BaseRiskStrategy):
    lookback = 20
    z_thresh = 1.5

    def init(self):
        super().init()
        self.mavg = self.I(sma, self.data.Close, self.lookback)
        self.std = self.I(lambda s, n: pd.Series(s).rolling(n).std().to_numpy(), self.data.Close, self.lookback)
        self.atr14 = self.I(atr, pd.DataFrame({
            "high": self.data.High, "low": self.data.Low, "close": self.data.Close
        }), 14)

    def next(self):
        self.check_stops()
        price = self.data.Close[-1]
        atr_val = float(self.atr14[-1]) if len(self.atr14) and not np.isnan(self.atr14[-1]) else None
        z = (price - self.mavg[-1]) / (self.std[-1] + 1e-9)
        if z < -self.z_thresh and not self.position.is_long:
            shares = self.size_from_equity(price, atr_val)
            if shares > 0:
                self.buy(size=shares)
                self.set_stops_on_entry(price, atr_val)
        elif z > self.z_thresh and self.position.is_long:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None


class RSIStrategy(BaseRiskStrategy):
    oversold = 30
    overbought = 70

    def init(self):
        super().init()
        self.rsi = self.I(rsi, self.data.Close, 14)
        self.atr14 = self.I(atr, pd.DataFrame({
            "high": self.data.High, "low": self.data.Low, "close": self.data.Close
        }), 14)

    def next(self):
        self.check_stops()
        price = self.data.Close[-1]
        atr_val = float(self.atr14[-1]) if len(self.atr14) and not np.isnan(self.atr14[-1]) else None
        if self.rsi[-1] < self.oversold and not self.position.is_long:
            shares = self.size_from_equity(price, atr_val)
            if shares > 0:
                self.buy(size=shares)
                self.set_stops_on_entry(price, atr_val)
        elif self.rsi[-1] > self.overbought and self.position.is_long:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None


class EnsembleStrategy(BaseRiskStrategy):
    def init(self):
        super().init()
        price = self.data.Close
        self.ma10 = self.I(sma, price, 10)
        self.ma20 = self.I(sma, price, 20)
        self.rsi = self.I(rsi, price, 14)
        self.mom20 = self.I(lambda x: pd.Series(x).pct_change(20).to_numpy(), price)
        self.atr14 = self.I(atr, pd.DataFrame({
            "high": self.data.High, "low": self.data.Low, "close": self.data.Close
        }), 14)

    def next(self):
        self.check_stops()
        votes = 0
        price = self.data.Close[-1]
        atr_val = float(self.atr14[-1]) if len(self.atr14) and not np.isnan(self.atr14[-1]) else None

        votes += 1 if self.ma10[-1] > self.ma20[-1] else -1
        votes += 1 if self.mom20[-1] > 0 else -1
        if self.rsi[-1] < 40: votes += 1
        elif self.rsi[-1] > 70: votes -= 1

        recent_mean = np.mean(self.data.Close[-20:])
        recent_std = np.std(self.data.Close[-20:])
        z = (price - recent_mean) / (recent_std + 1e-9)
        if z < -1.5: votes += 1
        elif z > 1.5: votes -= 1

        if votes > 0 and not self.position.is_long:
            shares = self.size_from_equity(price, atr_val)
            if shares > 0:
                self.buy(size=shares)
                self.set_stops_on_entry(price, atr_val)
        elif votes < 0 and self.position.is_long:
            self.position.close()
            self.entry_price = self.stop_price = self.take_price = None
