"""
core/strategy.py - Unified Long-Only Intraday Trading Strategy

This core strategy logic is used by both backtest engine and live trading bot.
It enforces long-only, intraday-only constraints while supporting diverse indicators
and configuration-driven parameterization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, Optional
from dataclasses import dataclass
import logging

from core.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """A trading signal with metadata."""
    action: str        # 'BUY', 'CLOSE', 'HOLD'
    timestamp: datetime
    price: float
    confidence: float = 1.0
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ModularIntradayStrategy:
    def __init__(self, params: Dict[str, any]):
        self.params = params
        self.name = "Modular Intraday Long-Only Strategy"
        self.version = "3.0"
        self.bars_processed = 0
        self.last_signal_time = None
        self.daily_stats = {'trades_today': 0, 'pnl_today': 0.0, 'last_trade_time': None, 'session_start_time': None}

        session_params = params.get('session', {})
        self.intraday_start = time(session_params.get('intraday_start_hour', 9),
                                  session_params.get('intraday_start_min', 15))
        self.intraday_end = time(session_params.get('intraday_end_hour', 15),
                                session_params.get('intraday_end_min', 15))
        self.exit_before_close = session_params.get('exit_before_close', 20)

        # Indicator toggles
        self.use_ema_crossover = params.get('use_ema_crossover', True)
        self.use_macd = params.get('use_macd', True)
        self.use_vwap = params.get('use_vwap', True)
        self.use_rsi_filter = params.get('use_rsi_filter', False)
        self.use_htf_trend = params.get('use_htf_trend', True)
        self.use_bollinger_bands = params.get('use_bollinger_bands', False)
        self.use_stochastic = params.get('use_stochastic', False)
        self.use_atr = params.get('use_atr', True)

        # EMA parameters
        self.fast_ema = params.get('fast_ema', 9)
        self.slow_ema = params.get('slow_ema', 21)
        self.ema_points_threshold = params.get('ema_points_threshold', 2)

        # MACD parameters
        self.macd_fast = params.get('macd_fast', 12)
        self.macd_slow = params.get('macd_slow', 26)
        self.macd_signal = params.get('macd_signal', 9)

        # RSI parameters
        self.rsi_length = params.get('rsi_length', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)

        # HTF parameters
        self.htf_period = params.get('htf_period', 20)

        # Risk management parameters
        self.base_sl_points = params.get('base_sl_points', 15)
        self.risk_per_trade_percent = params.get('risk_per_trade_percent', 1.0)
        self.max_positions_per_day = params.get('max_trades_per_day', 10)
        self.min_signal_gap = params.get('min_signal_gap_minutes', 5)
        self.no_trade_start_minutes = params.get('no_trade_start_minutes', 5)
        self.no_trade_end_minutes = params.get('no_trade_end_minutes', 30)

        logger.info(f"Strategy {self.name} v{self.version} initialized.")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculates all indicators required as per current config."""
        return calculate_all_indicators(data, self.params)

    def is_trading_session(self, now: datetime) -> bool:
        """Checks if current time is within trading session."""
        current_time = now.time()
        return self.intraday_start <= current_time <= self.intraday_end

    def should_exit_session(self, now: datetime) -> bool:
        """Checks if it is time to close positions before session end."""
        if not self.params.get('session', {}).get('is_intraday', True):
            return False
        session_end_dt = datetime.combine(now.date(), self.intraday_end)
        exit_time = session_end_dt - timedelta(minutes=self.exit_before_close)
        return now >= exit_time

    def can_enter_new_position(self, now: datetime) -> bool:
        """Checks whether new positions can be entered respecting session limits and trade count."""
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
        """Generates a long entry signal if all conditions meet."""
        if not self.can_enter_new_position(now):
            return TradingSignal('HOLD', now, row['close'], reason="Cannot enter new position")
        if self.bars_processed < 50:
            return TradingSignal('HOLD', now, row['close'], reason="Warming up indicators")

        conditions = []
        reasons = []

        # EMA Crossover Condition
        if self.use_ema_crossover:
            fast_ema = row.get('fast_ema', np.nan)
            slow_ema = row.get('slow_ema', np.nan)
            valid = not np.isnan(fast_ema) and not np.isnan(slow_ema) and (fast_ema - slow_ema >= self.ema_points_threshold)
            conditions.append(valid)
            reasons.append(f"EMA Cross {'passed' if valid else 'failed'}")

        # MACD Condition
        if self.use_macd:
            macd_bull = row.get('macd_bullish', False)
            macd_hist_pos = row.get('macd_histogram_positive', False)
            macd_valid = macd_bull and macd_hist_pos
            conditions.append(macd_valid)
            reasons.append(f"MACD {'passed' if macd_valid else 'failed'}")

        # VWAP Condition
        if self.use_vwap:
            vwap = row.get('vwap', np.nan)
            valid = not np.isnan(vwap) and (row['close'] > vwap)
            conditions.append(valid)
            reasons.append(f"VWAP {'passed' if valid else 'failed'}")

        # HTF Condition (optional)
        if self.use_htf_trend:
            htf_ema = row.get('htf_ema', np.nan)
            valid = not np.isnan(htf_ema) and (row['close'] > htf_ema)
            conditions.append(valid)
            reasons.append(f"HTF Trend {'passed' if valid else 'failed'}")

        # RSI Condition
        if self.use_rsi_filter:
            rsi = row.get('rsi', np.nan)
            valid = not np.isnan(rsi) and (self.rsi_oversold < rsi < self.rsi_overbought)
            conditions.append(valid)
            reasons.append(f"RSI {'passed' if valid else 'failed'}")

        # Bollinger Bands Condition
        if self.use_bollinger_bands:
            bb_lower = row.get('bb_lower', np.nan)
            bb_upper = row.get('bb_upper', np.nan)
            valid = not np.isnan(bb_lower) and not np.isnan(bb_upper) and (bb_lower < row['close'] < bb_upper)
            conditions.append(valid)
            reasons.append(f"Bollinger Bands {'passed' if valid else 'failed'}")

        if all(conditions):
            self.last_signal_time = now
            stop_loss = row['close'] - self.base_sl_points
            return TradingSignal('BUY', now, row['close'], reason="; ".join(reasons), stop_loss=stop_loss)
        failed_reasons = [r for c, r in zip(conditions, reasons) if not c]
        return TradingSignal('HOLD', now, row['close'], confidence=0.0, reason="Blocked: " + "; ".join(failed_reasons[:3]))

    def should_enter_long(self, row: pd.Series, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now()
        sig = self.generate_entry_signal(row, now)
        return sig.action == 'BUY'

    def should_enter_short(self, row: pd.Series, now: Optional[datetime] = None) -> bool:
        return False  # Long-only strategy

    def should_exit_position(self, row: pd.Series, position_type: str, now: Optional[datetime] = None) -> bool:
        if now is None:
            now = datetime.now()
        return self.should_exit_session(now)

    def open_long(self, row: pd.Series, now: datetime, position_manager) -> Optional[str]:
        try:
            position_id = position_manager.open_position(
                symbol=self.params.get('symbol', 'NIFTY'),
                entry_price=row['close'],
                timestamp=now,
                lot_size=self.params.get('lot_size', 1),
                tick_size=self.params.get('tick_size', 0.05)
            )
            if position_id:
                self.daily_stats['trades_today'] += 1
                self.last_signal_time = now
                logger.info(f"Opened long position {position_id} @ {row['close']}")
            return position_id
        except Exception as e:
            logger.error(f"Error opening long: {e}")
            return None

    def handle_exit(self, position_id: str, price: float, now: datetime, position_manager, reason: str = "Strategy Exit"):
        try:
            position_manager.close_position_full(position_id, price, now, reason)
            logger.info(f"Closed position {position_id} at {price} reason: {reason}")
        except Exception as e:
            logger.error(f"Error closing position: {e}")

    def on_new_bar(self, row: pd.Series, now: datetime):
        self.bars_processed += 1
        if (self.daily_stats['session_start_time'] is None or now.date() != self.daily_stats['session_start_time'].date()):
            self.daily_stats = {'trades_today':0, 'pnl_today':0.0, 'last_trade_time': None, 'session_start_time': now}

    def reset(self):
        self.bars_processed = 0
        self.last_signal_time = None
        self.daily_stats = {'trades_today':0, 'pnl_today':0.0, 'last_trade_time': None, 'session_start_time': None}
        logger.info(f"{self.name} reset")

    def validate_parameters(self) -> list:
        errors = []
        if self.fast_ema >= self.slow_ema:
            errors.append("Fast EMA must be less than Slow EMA.")
        if self.risk_per_trade_percent <= 0:
            errors.append("Risk per trade must be positive.")
        if self.base_sl_points <= 0:
            errors.append("Base stop loss must be positive.")
        if self.intraday_start >= self.intraday_end:
            errors.append("Session start time must be before session end time.")
        return errors

    def get_strategy_info(self) -> Dict[str, any]:
        return {
            'name': self.name,
            'version': self.version,
            'type': 'Long-Only Intraday',
            'indicators_enabled': {
                'ema_crossover': self.use_ema_crossover,
                'macd': self.use_macd,
                'vwap': self.use_vwap,
                'rsi_filter': self.use_rsi_filter,
                'htf_trend': self.use_htf_trend,
                'bollinger_bands': self.use_bollinger_bands,
                'stochastic': self.use_stochastic,
                'atr': self.use_atr,
            },
            'parameters': {
                'fast_ema': self.fast_ema,
                'slow_ema': self.slow_ema,
                'ema_points_threshold': self.ema_points_threshold,
                'htf_period': self.htf_period,
                'base_sl_points': self.base_sl_points,
                'risk_per_trade_percent': self.risk_per_trade_percent,
            },
            'constraints': {
                'long_only': True,
                'intraday_only': True,
                'max_trades_per_day': self.max_positions_per_day,
            },
            'session': {
                'start': self.intraday_start.strftime("%H:%M"),
                'end': self.intraday_end.strftime("%H:%M"),
                'exit_before_close': self.exit_before_close,
            },
            'daily_stats': self.daily_stats.copy(),
        }
