"""
core/strategy.py - Unified Long-Only, Intraday Strategy for Trading Bot and Backtest

- F&O-ready, multi-indicator, live-driven.
- No shorting, no overnight risk, all config/param driven.
- Handles all signal, entry, exit, and session rules.
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, time, timedelta
from utils.time_utils import now_ist, normalize_datetime_to_ist, is_time_to_exit

from core.indicators import IncrementalEMA, IncrementalMACD, IncrementalVWAP, IncrementalATR

class ModularIntradayStrategy:
    def __init__(self, config: Dict[str, Any], indicators_module):
        self.config = config
        self.indicators = indicators_module
        self.in_position = False
        self.position_id = None
        self.position_entry_time = None
        self.position_entry_price = None
        self.last_signal_time = None
        # For metering daily trade count and other constraints
        self.daily_trade_count = 0

        # Session/session exit config
        session = config.get('session', {})
        self.intraday_start = time(session.get('intraday_start_hour', 9), session.get('intraday_start_min', 15))
        self.intraday_end = time(session.get('intraday_end_hour', 15), session.get('intraday_end_min', 30))
        self.exit_before_close = session.get('exit_before_close', 20)
        self.max_trades_per_day = config.get('max_trades_per_day',25)

        # --- Incremental indicator trackers ---    
        self.ema_fast_tracker = IncrementalEMA(period=self.config.get('fast_ema', 9))
        self.ema_slow_tracker = IncrementalEMA(period=self.config.get('slow_ema', 21))
        self.macd_tracker = IncrementalMACD(
            fast=self.config.get('macd_fast', 12),
            slow=self.config.get('macd_slow', 26),
            signal=self.config.get('macd_signal', 9)
        )
        self.vwap_tracker = IncrementalVWAP()
        self.atr_tracker = IncrementalATR(period=self.config.get('atr_len', 14))

    def is_session_live(self, current_time: datetime) -> bool:
        t = current_time.time()
        return self.intraday_start <= t <= self.intraday_end

    def should_exit_for_session(self, now: datetime) -> bool:
        """Enhanced session exit logic with user-configurable buffer"""
        session = self.config.get('session', {})
        exit_buffer = session.get('exit_before_close', 20)
        end_hour = session.get('intraday_end_hour', 15)
        end_min = session.get('intraday_end_min', 30)
        # Use centralized session exit logic
        from utils.time_utils import is_time_to_exit
        return is_time_to_exit(now, exit_buffer, end_hour, end_min)

    def is_market_closed(self, current_time: datetime) -> bool:
        """Check if market is completely closed (after end time)"""
        session = self.config.get('session', {})
        end_hour = session.get('intraday_end_hour', 15)
        end_min = session.get('intraday_end_min', 30)
        current_minutes = current_time.hour * 60 + current_time.minute
        end_minutes = end_hour * 60 + end_min
        return current_minutes >= end_minutes

    def indicators_and_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all active indicators using the correct module (live or backtest-safe)."""
        return self.indicators.calculate_all_indicators(data, self.config)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for the strategy; memory optim parameters removed."""
        return self.indicators_and_signals(data)

    def entry_signal(self, row: pd.Series) -> bool:
        # --- EMA CROSS ---
        pass_ema = True
        if self.config.get('use_ema_crossover', False):
            pass_ema = (
                row.get('fast_ema', None) is not None and
                row.get('slow_ema', None) is not None and
                row['fast_ema'] > row['slow_ema']
            )
        # --- VWAP ---
        pass_vwap = True
        if self.config.get('use_vwap', False):
            vwap_val = row.get('vwap', None)
            pass_vwap = (vwap_val is not None) and (row['close'] > vwap_val)
        # --- MACD ---
        pass_macd = True
        if self.config.get('use_macd', False):
            pass_macd = row.get('macd_bullish', False) and row.get('macd_histogram_positive', False)
        # --- HTF TREND ---
        pass_htf = True
        if self.config.get('use_htf_trend', False):
            htf_val = row.get('htf_ema', None)
            pass_htf = (htf_val is not None) and (row['close'] > htf_val)
        # --- RSI ---
        pass_rsi = True
        if self.config.get('use_rsi_filter', False):
            rsi_val = row.get('rsi', None)
            pass_rsi = (rsi_val is not None) and (self.config.get('rsi_oversold', 30) < rsi_val < self.config.get('rsi_overbought', 70))
        # --- Bollinger Bands ---
        pass_bb = True
        if self.config.get('use_bollinger_bands', False):
            bb_lower = row.get('bb_lower', None)
            bb_upper = row.get('bb_upper', None)
            pass_bb = (bb_lower is not None and bb_upper is not None) and (bb_lower < row['close'] < bb_upper)
        # --- Construct final pass signal (all enabled must be True) ---
        logic_checks = [pass_ema, pass_vwap, pass_macd, pass_htf, pass_rsi, pass_bb]
        return all(logic_checks)

    def can_open_long(self, row: pd.Series, now: datetime) -> bool:
        # Only for live session, long-only; never more than max_trades_per_day.
        if not self.is_session_live(now):
            return False
        if self.daily_trade_count >= self.max_trades_per_day:
            return False
        if self.in_position:
            return False
        if not self.entry_signal(row):
            return False
        if self.should_exit_for_session(now):
            return False
        return True

    def open_long(self, row: pd.Series, now: datetime, position_manager) -> Optional[str]:
        # For robust trade management, always use live/production-driven position config
        price = row['close']
        symbol = self.config.get('symbol', 'N/A')
        # lot size and tick size must be passed from config (for F&O)
        lot_size = self.config.get('lot_size', 1)
        tick_size = self.config.get('tick_size', 0.05)
        pos_id = position_manager.open_position(
            symbol, price, now, lot_size=lot_size, tick_size=tick_size
        )
        if pos_id:
            self.in_position = True
            self.position_id = pos_id
            self.position_entry_time = now
            self.position_entry_price = price
            self.daily_trade_count += 1
            self.last_signal_time = now
            return pos_id
        return None

    def should_close(self, row: pd.Series, now: datetime, position_manager) -> bool:
        # Always flatten before session end, or let position manager enforce stops/TP/trail
        return self.should_exit_for_session(now)

    def handle_exit(self, position_id: str, price: float, now: datetime, position_manager, reason="Session End"):
        if not position_id:
            return
        position_manager.close_position_full(position_id, price, now, reason=reason)
        self.in_position = False
        self.position_id = None
        self.position_entry_time = None
        self.position_entry_price = None

    def reset_daily_counters(self, now: datetime):
        # Should be called at new session start
        self.daily_trade_count = 0
        self.last_signal_time = None

    def validate_parameters(self) -> list:
        errors = []
        # Typical validation rules
        if self.config.get('use_ema_crossover', False):
            if self.config['fast_ema'] >= self.config['slow_ema']:
                errors.append("fast_ema must be less than slow_ema")
        if self.config.get('use_htf_trend', False):
            if self.config['htf_period'] <= 0:
                errors.append("htf_period must be positive")
        return errors

    def process_tick_or_bar(self, row: pd.Series):
        # For EMA
        fast_ema_val = self.ema_fast_tracker.update(row['close'])
        slow_ema_val = self.ema_slow_tracker.update(row['close'])

        # For MACD
        macd_val, macd_signal_val, macd_hist_val = self.macd_tracker.update(row['close'])

        # For VWAP
        vwap_val = self.vwap_tracker.update(
            price=row['close'], volume=row['volume'],
            high=row.get('high'), low=row.get('low'), close=row.get('close')
        )

        # For ATR
        atr_val = self.atr_tracker.update(
            high=row['high'], low=row['low'], close=row['close']
        )

        # Update row/signal state as required
        row['fast_ema'] = fast_ema_val
        row['slow_ema'] = slow_ema_val
        row['macd'] = macd_val
        row['macd_signal'] = macd_signal_val
        row['macd_histogram'] = macd_hist_val
        row['vwap'] = vwap_val
        row['atr'] = atr_val

if __name__ == "__main__":
    # Minimal smoke test for development
    test_params = {
        'use_ema_crossover': True, 'fast_ema': 9, 'slow_ema': 21, 'ema_points_threshold': 2,
        'use_macd': True, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
        'use_vwap': True, 'use_htf_trend': True, 'htf_period': 20, 'symbol': 'NIFTY24DECFUT', 'lot_size': 15, 'tick_size': 0.05,
        'session': {'intraday_start_hour': 9, 'intraday_start_min': 15, 'intraday_end_hour': 15, 'intraday_end_min': 30, 'exit_before_close': 20},
        'max_trades_per_day': 25
    }
    import core.indicators as indicators
    strat = ModularIntradayStrategy(test_params, indicators)
    print("Parameter validation errors:", strat.validate_parameters())

"""
CONFIGURATION PARAMETER NAMING CONVENTION:
- Constructor parameter: __init__(self, config: Dict[str, Any], ...)
- Internal storage: self.config = config
- All parameter access: self.config.get('parameter_name', default)
- Session parameters extracted to dedicated variables in constructor

INTERFACE CONSISTENCY:
- Uses same parameter naming as researchStrategy.py
- calculate_indicators() passes self.config to indicators module
- Compatible with both backtest and live trading systems

CRITICAL: This file must maintain naming consistency with researchStrategy.py
- Both use 'config' parameter in constructor
- Both use self.config for internal parameter access
- Both pass self.config to calculate_all_indicators()
"""
