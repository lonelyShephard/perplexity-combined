"""
core/researchStrategy.py

Unified long-only, intraday research strategy supporting:
- Multiple indicator toggles (EMA, MACD, VWAP, HTF, RSI, BB, ATR, etc.)
- Parameter-driven configuration and constraints
- Strict session, position, and entry/exit management
- Full compatibility with both backtest and forward test modules
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from dataclasses import dataclass
import logging
from core.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    action: str        # 'BUY', 'CLOSE', 'HOLD'
    timestamp: datetime
    price: float
    confidence: float = 1.0
    reason: str = ""
    stop_loss: float = None
    take_profit: float = None

class ModularIntradayStrategy:
    def __init__(self, params: dict, indicators_mod=None):
        self.params = params
        self.name = "Research Intraday Long-Only Strategy"
        self.version = "R1"
        self.last_signal_time = None
        self.daily_stats = {"trades_today": 0, "last_trade_time": None}
        self.min_bars_required = 50

        session = params.get("session", {})
        self.intraday_start = time(session.get("intraday_start_hour", 9), session.get("intraday_start_min", 15))
        self.intraday_end = time(session.get("intraday_end_hour", 15), session.get("intraday_end_min", 15))
        self.exit_before_close = session.get("exit_before_close", 20)
        self.max_positions_per_day = params.get("max_trades_per_day", 10)
        self.no_trade_start_minutes = params.get("no_trade_start_minutes", 5)
        self.no_trade_end_minutes = params.get("no_trade_end_minutes", 30)
        self.min_signal_gap = params.get("min_signal_gap_minutes", 5)

        # Indicator toggles
        self.use_ema_crossover = params.get("use_ema_crossover", True)
        self.use_macd = params.get("use_macd", True)
        self.use_vwap = params.get("use_vwap", True)
        self.use_rsi_filter = params.get("use_rsi_filter", False)
        self.use_htf_trend = params.get("use_htf_trend", True)
        self.use_bollinger_bands = params.get("use_bollinger_bands", False)
        self.use_stochastic = params.get("use_stochastic", False)
        self.use_ma = params.get("use_ma", False)
        self.use_atr = params.get("use_atr", True)
        self.base_sl_points = params.get("base_sl_points", 15)
        self.ema_points_threshold = params.get("ema_points_threshold", 2)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute all enabled indicators and signals."""
        return calculate_all_indicators(data, self.params)

    def is_trading_session(self, now: datetime) -> bool:
        current_time = now.time()
        return self.intraday_start <= current_time <= self.intraday_end

    def should_exit_session(self, now: datetime) -> bool:
        if not self.params.get("session", {}).get("is_intraday", True):
            return False
        session_end = datetime.combine(now.date(), self.intraday_end)
        exit_time = session_end - timedelta(minutes=self.exit_before_close)
        return now >= exit_time

    def can_enter_new_position(self, now: datetime) -> bool:
        if not self.is_trading_session(now):
            return False
        if self.daily_stats['trades_today'] >= self.max_positions_per_day:
            return False
        session_start = datetime.combine(now.date(), self.intraday_start)
        session_end = datetime.combine(now.date(), self.intraday_end)
        if now < session_start + timedelta(minutes=self.no_trade_start_minutes):
            return False
        if now > session_end - timedelta(minutes=self.no_trade_end_minutes):
            return False
        if self.last_signal_time is not None:
            gap_min = (now - self.last_signal_time).total_seconds() / 60
            if gap_min < self.min_signal_gap:
                return False
        return True

    def generate_entry_signal(self, row: pd.Series, now: datetime) -> TradingSignal:
        if not self.can_enter_new_position(now):
            return TradingSignal('HOLD', now, row['close'], reason="Blocked by session/rules")
        indicators_ready = row.notnull().sum() > self.min_bars_required
        if not indicators_ready:
            return TradingSignal('HOLD', now, row['close'], reason="Indicator warmup")

        conditions = []
        reasons = []

        if self.use_ema_crossover:
            valid = row.get("fast_ema", np.nan) - row.get("slow_ema", np.nan) >= self.ema_points_threshold
            conditions.append(valid)
            reasons.append(f"EMA {'PASS' if valid else 'FAIL'}")

        if self.use_macd:
            macd = row.get("macd_bullish", False) and row.get("macd_histogram_positive", False)
            conditions.append(macd)
            reasons.append(f"MACD {'PASS' if macd else 'FAIL'}")
        
        if self.use_vwap:
            vwap_ok = row.get("close", np.nan) > row.get("vwap", np.nan)
            conditions.append(vwap_ok)
            reasons.append(f"VWAP {'PASS' if vwap_ok else 'FAIL'}")

        if self.use_htf_trend:
            htf_ok = row.get("close", np.nan) > row.get("htf_ema", np.nan)
            conditions.append(htf_ok)
            reasons.append(f"HTF {'PASS' if htf_ok else 'FAIL'}")

        if self.use_rsi_filter:
            rsi = row.get("rsi", np.nan)
            rsi_ok = (rsi > self.params.get("rsi_oversold", 30)) and (rsi < self.params.get("rsi_overbought", 70))
            conditions.append(rsi_ok)
            reasons.append(f"RSI {'PASS' if rsi_ok else 'FAIL'}")

        if self.use_bollinger_bands:
            bb_ok = row.get("bb_lower", np.nan) < row.get("close", np.nan) < row.get("bb_upper", np.nan)
            conditions.append(bb_ok)
            reasons.append(f"BB {'PASS' if bb_ok else 'FAIL'}")

        if all(conditions):
            self.last_signal_time = now
            stop_loss = row['close'] - self.base_sl_points
            return TradingSignal('BUY', now, row['close'], reason="Ok: " + ", ".join(reasons), stop_loss=stop_loss)
        blocked = [r for (c, r) in zip(conditions, reasons) if not c]
        return TradingSignal('HOLD', now, row['close'], reason="Blocked: " + ", ".join(blocked[:3]))

    def should_enter_long(self, row: pd.Series, now: datetime = None) -> bool:
        now = now or datetime.now()
        signal = self.generate_entry_signal(row, now)
        return signal.action == "BUY"

    def open_long(self, row: pd.Series, now: datetime, position_manager) -> str:
        position_id = position_manager.open_position(
            symbol=self.params.get("symbol", "NIFTY"),
            entry_price=row['close'],
            timestamp=now,
            lot_size=self.params.get("lot_size", 1),
            tick_size=self.params.get("tick_size", 0.05)
        )
        if position_id:
            self.daily_stats['trades_today'] += 1
            self.last_signal_time = now
            logger.info(f"NEW LONG {position_id} @ {row['close']:.2f}")
        return position_id

    def should_exit_position(self, row: pd.Series, position_type: str, now: datetime = None) -> bool:
        return self.should_exit_session(now or datetime.now())

    def should_close(self, row: pd.Series, now: datetime, position_manager) -> bool:
        return self.should_exit_session(now)

    def handle_exit(self, position_id: str, price: float, now: datetime, position_manager, reason: str = "Strategy Exit"):
        position_manager.close_position_full(position_id, price, now, reason)
        logger.info(f"CLOSED {position_id} at {price:.2f} ({reason})")

    # For compatibility
    def can_open_long(self, row: pd.Series, now: datetime) -> bool:
        return self.should_enter_long(row, now)

    def indicators_and_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy compatibility: calculate indicators and signals."""
        return self.calculate_indicators(df)
