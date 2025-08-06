"""
core/strategy.py - Unified Long-Only Intraday Trading Strategy

This is the core strategy logic used by both backtest engine and live trading bot.
Enforces long-only, intraday-only constraints while supporting all indicators.

Features:
- Multi-indicator signal generation
- Parameter-driven configuration  
- Long-only, intraday-only enforcement
- Session management
- Entry/exit logic with multiple filters
- Compatible with both backtest and live systems
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import pytz
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Import indicators
from core.indicators import calculate_all_indicators
from core.indicators import IncrementalEMA, IncrementalMACD, IncrementalVWAP, IncrementalATR
from utils.time_utils import now_ist, normalize_datetime_to_ist, is_time_to_exit, ensure_tz_aware


logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Represents a trading signal with all necessary information."""
    action: str  # 'BUY', 'CLOSE', 'HOLD'
    timestamp: datetime
    price: float
    confidence: float = 1.0
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class ModularIntradayStrategy:
    """
    Unified long-only intraday strategy supporting multiple indicators.
    """
    
    def __init__(self, config: Dict[str, Any], indicators_module=None):
        """
        Initialize strategy with parameters.
        
        Args:
            config: Strategy parameters from config
        """
        self.config = config
        self.indicators = indicators_module  # Store but don't necessarily use
        self.name = "Modular Intraday Long-Only Strategy"
        self.version = "3.0"
        
        # Strategy state
        self.is_initialized = False
        self.current_position = None
        self.last_signal_time = None
        self.min_bars_required = 50  # Minimum bars needed for indicators
        self.bars_processed = 0
        
        # Session management
        self.session_params = config.get('session', {})
        self.intraday_start = time(
            self.session_params.get('intraday_start_hour', 9),
            self.session_params.get('intraday_start_min', 15)
        )
        self.intraday_end = time(
            self.session_params.get('intraday_end_hour', 15),
            self.session_params.get('intraday_end_min', 30)
        )
        self.exit_before_close = self.session_params.get('exit_before_close', 20)
        
        # Trading constraints
        self.max_positions_per_day = config.get('max_trades_per_day', 10)
        self.min_signal_gap = config.get('min_signal_gap_minutes', 5)
        self.no_trade_start_minutes = config.get('no_trade_start_minutes', 5)
        self.no_trade_end_minutes = config.get('no_trade_end_minutes', 30)
        
        # Indicator parameters
        self.use_ema_crossover = config.get('use_ema_crossover', True)
        self.use_macd = config.get('use_macd', True)
        self.use_vwap = config.get('use_vwap', True)
        self.use_rsi_filter = config.get('use_rsi_filter', False)
        self.use_htf_trend = config.get('use_htf_trend', True)  # Now optional!
        self.use_bollinger_bands = config.get('use_bollinger_bands', False)
        self.use_stochastic = config.get('use_stochastic', False)
        self.use_atr = config.get('use_atr', True)
        
        # EMA parameters
        self.fast_ema = config.get('fast_ema', 9)
        self.slow_ema = config.get('slow_ema', 21)
 
        # MACD parameters
        self.macd_fast = config.get('macd_fast', 12)
        self.macd_slow = config.get('macd_slow', 26)
        self.macd_signal = config.get('macd_signal', 9)
        
        # RSI parameters
        self.rsi_length = config.get('rsi_length', 14)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        
        # HTF parameters
        self.htf_period = config.get('htf_period', 20)
        
        # Risk management
        self.base_sl_points = config.get('base_sl_points', 15)
        self.risk_per_trade_percent = config.get('risk_per_trade_percent', 1.0)
        
        # Daily tracking
        self.daily_stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'last_trade_time': None,
            'session_start_time': None
        }

        # --- Incremental indicator trackers ---
        self.ema_fast_tracker = IncrementalEMA(period=self.fast_ema)
        self.ema_slow_tracker = IncrementalEMA(period=self.slow_ema)
        self.macd_tracker = IncrementalMACD(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        self.vwap_tracker = IncrementalVWAP()
        self.atr_tracker = IncrementalATR(period=self.config.get('atr_len', 14))

        logger.info(f"Strategy initialized: {self.name} v{self.version}")
        logger.info(f"Indicators enabled: EMA={self.use_ema_crossover}, MACD={self.use_macd}, "
                   f"VWAP={self.use_vwap}, HTF={self.use_htf_trend}, RSI={self.use_rsi_filter}")
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators."""
        return calculate_all_indicators(df, self.config)
    
    def is_trading_session(self, current_time: datetime) -> bool:
        """
        Check if current time is within trading session.
        """
        # Convert timezone-aware datetime to naive time for comparison
        # This ensures consistent comparison regardless of timezone info
        if current_time.tzinfo is not None:
            # Create local time for comparison
            local_time = current_time.time()
            # If the datetime is timezone-aware, we need to compare the hour and minute directly
            return (self.intraday_start.hour <= local_time.hour <= self.intraday_end.hour and 
                   (local_time.hour > self.intraday_start.hour or local_time.minute >= self.intraday_start.minute) and
                   (local_time.hour < self.intraday_end.hour or local_time.minute <= self.intraday_end.minute))
        else:
            # Original logic for naive datetimes
            t = current_time.time()
            return self.intraday_start <= t <= self.intraday_end
    
    def should_exit_for_session(self, now: datetime) -> bool:
        """Enhanced session exit logic with user-configurable buffer"""
        if not hasattr(self, 'session_params'):
            return False
            
        exit_buffer = self.session_params.get('exit_before_close', 20)
        end_hour = self.session_params.get('intraday_end_hour', 15)
        end_min = self.session_params.get('intraday_end_min', 30)
        
        from utils.time_utils import is_time_to_exit
        return is_time_to_exit(now, exit_buffer, end_hour, end_min)

    def is_market_closed(self, current_time: datetime) -> bool:
        """Check if market is completely closed (after end time)"""
        if not hasattr(self, 'session_params'):
            return False
            
        end_hour = self.session_params.get('intraday_end_hour', 15)
        end_min = self.session_params.get('intraday_end_min', 30)
        
        current_minutes = current_time.hour * 60 + current_time.minute
        end_minutes = end_hour * 60 + end_min
        
        return current_minutes >= end_minutes
    
    def can_enter_new_position(self, current_time: datetime) -> bool:
        """
        Check if new positions can be entered.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if can enter new position
        """
        # Check if in trading session
        if not self.is_trading_session(current_time):
            return False
        
        # Check daily trade limit
        if self.daily_stats['trades_today'] >= self.max_positions_per_day:
            return False
        
        # Check no-trade periods
        session_start = datetime.combine(current_time.date(), self.intraday_start)
        session_end = datetime.combine(current_time.date(), self.intraday_end)

        # Ensure session_start and session_end are timezone-aware
        session_start = ensure_tz_aware(session_start, current_time.tzinfo)
        session_end = ensure_tz_aware(session_end, current_time.tzinfo)
        
        # No trades in first few minutes
        if current_time < session_start + timedelta(minutes=self.no_trade_start_minutes):
            return False
            
        # No trades in last few minutes
        if current_time > session_end - timedelta(minutes=self.no_trade_end_minutes):
            return False
        
        # Check minimum gap between trades
        if self.last_signal_time:
            time_gap = (current_time - self.last_signal_time).total_seconds() / 60
            if time_gap < self.min_signal_gap:
                return False
        
        return True
    
    def generate_entry_signal(self, row: pd.Series, current_time: datetime) -> TradingSignal:
        """
        Generate entry signal based on all enabled indicators.
        
        Args:
            row: Current data row with indicators
            current_time: Current timestamp
            
        Returns:
            TradingSignal object
        """
        # Check if we can enter
        if not self.can_enter_new_position(current_time):
            return TradingSignal('HOLD', current_time, row['close'], reason="Cannot enter new position")
        
        # Check if minimum bars processed
        if self.bars_processed < self.min_bars_required:
            return TradingSignal('HOLD', current_time, row['close'], reason="Warming up indicators")
        
        # Collect all signal conditions
        signal_conditions = []
        signal_reasons = []
        confidence = 1.0
        
        # === EMA CROSSOVER SIGNAL ===
        if self.use_ema_crossover:
            if ('fast_ema' in row and 'slow_ema' in row and
                not pd.isna(row['fast_ema']) and not pd.isna(row['slow_ema'])):
                # Check EMA crossover 
                fast_ema = row['fast_ema']
                slow_ema = row['slow_ema']
                
                # Add detailed logging every 1000 calls
                if not hasattr(self, '_ema_log_count'):
                    self._ema_log_count = 0
                self._ema_log_count += 1
                
                if self._ema_log_count % 1000 == 0:
                    logger.info(f"EMA Analysis #{self._ema_log_count//1000}:")
                    logger.info(f"  Fast EMA: {fast_ema:.4f}")
                    logger.info(f"  Slow EMA: {slow_ema:.4f}")
                    logger.info(f"  Difference: {fast_ema - slow_ema:.4f}")
                    logger.info(f"  Bullish: {fast_ema > slow_ema}")
                
                if fast_ema > slow_ema:
                    signal_conditions.append(True)
                    signal_reasons.append(f"EMA Cross: {fast_ema:.2f} > {slow_ema:.2f}")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"EMA Cross: Fast EMA not above Slow EMA")
            else:
                signal_conditions.append(False)
                signal_reasons.append("EMA Cross: Data not available")
        
        # === MACD SIGNAL ===
        if self.use_macd:
            if ('macd_bullish' in row and 'macd_histogram_positive' in row):
                macd_bullish = row.get('macd_bullish', False)
                histogram_positive = row.get('macd_histogram_positive', False)
                
                # MACD bullish: MACD line > signal line AND histogram > 0
                macd_signal = macd_bullish and histogram_positive
                signal_conditions.append(macd_signal)
                
                if macd_signal:
                    signal_reasons.append("MACD: Bullish (line > signal & histogram > 0)")
                else:
                    signal_reasons.append(f"MACD: Not bullish (line>{row.get('macd_signal', 'NA')}: {macd_bullish}, hist>0: {histogram_positive})")
            else:
                signal_conditions.append(False)
                signal_reasons.append("MACD: Data not available")
        
        # === VWAP SIGNAL ===
        if self.use_vwap:
            if 'vwap' in row and not pd.isna(row['vwap']):
                vwap_bullish = row['close'] > row['vwap']
                signal_conditions.append(vwap_bullish)
                
                if vwap_bullish:
                    signal_reasons.append(f"VWAP: Bullish ({row['close']:.2f} > {row['vwap']:.2f})")
                else:
                    signal_reasons.append(f"VWAP: Bearish ({row['close']:.2f} <= {row['vwap']:.2f})")
            else:
                signal_conditions.append(False)
                signal_reasons.append("VWAP: Data not available")
        
        # === HTF TREND SIGNAL (Now Optional!) ===
        if self.use_htf_trend:
            if 'htf_ema' in row and not pd.isna(row['htf_ema']):
                htf_bullish = row['close'] > row['htf_ema']
                signal_conditions.append(htf_bullish)
                
                if htf_bullish:
                    signal_reasons.append(f"HTF Trend: Bullish ({row['close']:.2f} > {row['htf_ema']:.2f})")
                else:
                    signal_reasons.append(f"HTF Trend: Bearish ({row['close']:.2f} <= {row['htf_ema']:.2f})")
            else:
                signal_conditions.append(False)
                signal_reasons.append("HTF Trend: Data not available")
        
        # === RSI FILTER ===
        if self.use_rsi_filter:
            if 'rsi' in row and not pd.isna(row['rsi']):
                rsi = row['rsi']
                # RSI should be between oversold and overbought for entry
                rsi_ok = self.rsi_oversold < rsi < self.rsi_overbought
                signal_conditions.append(rsi_ok)
                
                if rsi_ok:
                    signal_reasons.append(f"RSI: Neutral ({rsi:.1f})")
                else:
                    signal_reasons.append(f"RSI: Extreme ({rsi:.1f})")
            else:
                signal_conditions.append(False)
                signal_reasons.append("RSI: Data not available")
        
        # === BOLLINGER BANDS FILTER ===
        if self.use_bollinger_bands:
            if all(col in row for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                bb_ok = row['bb_lower'] < row['close'] < row['bb_upper']
                signal_conditions.append(bb_ok)
                
                if bb_ok:
                    signal_reasons.append("BB: Price within bands")
                else:
                    signal_reasons.append("BB: Price outside bands")
            else:
                signal_conditions.append(False)
                signal_reasons.append("Bollinger Bands: Data not available")
        
        # === FINAL SIGNAL DECISION ===
        # ALL enabled conditions must be True for BUY signal
        if signal_conditions and all(signal_conditions):
            # Calculate stop loss
            stop_loss_price = row['close'] - self.base_sl_points
            
            # Update tracking
            self.last_signal_time = current_time
            
            return TradingSignal(
                action='BUY',
                timestamp=current_time,
                price=row['close'],
                confidence=confidence,
                reason="; ".join(signal_reasons),
                stop_loss=stop_loss_price
            )
        else:
            # Log why signal failed
            failed_reasons = [reason for i, reason in enumerate(signal_reasons) 
                            if i < len(signal_conditions) and not signal_conditions[i]]
            
            return TradingSignal(
                action='HOLD',
                timestamp=current_time,
                price=row['close'],
                confidence=0.0,
                reason=f"Entry blocked: {'; '.join(failed_reasons[:3])}"  # Limit message length
            )
    
    def should_enter_long(self, row: pd.Series, current_time: Optional[datetime] = None) -> bool:
        """
        Check if should enter long position (for backtest compatibility).
        
        Args:
            row: Current data row
            current_time: Current timestamp
            
        Returns:
            True if should enter long
        """
        if current_time is None:
            current_time = row.name if hasattr(row, 'name') else datetime.now()
        
        signal = self.generate_entry_signal(row, current_time)
        return signal.action == 'BUY'
    
    def should_enter_short(self, row: pd.Series, current_time: Optional[datetime] = None) -> bool:
        """
        Check if should enter short position.
        
        This strategy is LONG-ONLY, so this always returns False.
        
        Returns:
            False (no short positions allowed)
        """
        return False  # Long-only strategy
    
    def should_exit_position(self, row: pd.Series, position_type: str, 
                           current_time: Optional[datetime] = None) -> bool:
        """
        Check if should exit current position (for backtest compatibility).
        
        Args:
            row: Current data row
            position_type: Position type ('long' or 'short')
            current_time: Current timestamp
            
        Returns:
            True if should exit position
        """
        if current_time is None:
            current_time = row.name if hasattr(row, 'name') else datetime.now()
        
        # Always exit at session end
        if self.should_exit_session(current_time):
            return True
        
        # Let position manager handle stop loss, take profit, and trailing stops
        return False
    
    def get_signal_description(self, row: pd.Series) -> str:
        """
        Get human-readable signal description (for backtest compatibility).
        
        Args:
            row: Current data row
            
        Returns:
            Signal description string
        """
        descriptions = []
        
        if self.use_ema_crossover and 'fast_ema' in row and 'slow_ema' in row:
            fast_ema = row['fast_ema']
            slow_ema = row['slow_ema']
            if not pd.isna(fast_ema) and not pd.isna(slow_ema):
                descriptions.append(f"EMA {self.fast_ema}/{self.slow_ema}: {fast_ema:.2f}/{slow_ema:.2f}")
        
        if self.use_macd and 'macd' in row and 'macd_signal' in row:
            macd = row['macd']
            signal = row['macd_signal']
            if not pd.isna(macd) and not pd.isna(signal):
                descriptions.append(f"MACD: {macd:.3f}/{signal:.3f}")
        
        if self.use_vwap and 'vwap' in row:
            vwap = row['vwap']
            if not pd.isna(vwap):
                descriptions.append(f"VWAP: {row['close']:.2f} vs {vwap:.2f}")
        
        if self.use_htf_trend and 'htf_ema' in row:
            htf_ema = row['htf_ema']
            if not pd.isna(htf_ema):
                descriptions.append(f"HTF: {row['close']:.2f} vs {htf_ema:.2f}")
        
        return "; ".join(descriptions) if descriptions else "No indicators"
    
    def verify_backtest_interface(self):
        """Production verification of backtest interface."""
        required_methods = ['can_open_long', 'open_long', 'calculate_indicators', 'should_close']
        
        for method in required_methods:
            if not hasattr(self, method):
                logger.error(f"MISSING METHOD: {method}")
                return False
            else:
                logger.info(f"✅ Method exists: {method}")
        
        return True

    def can_open_long(self, row: pd.Series, timestamp: datetime) -> bool:
        """PRODUCTION INTERFACE: Entry signal detection."""
        try:
            # Ensure timezone awareness
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize('Asia/Kolkata')
            elif timestamp.tzinfo != pytz.timezone('Asia/Kolkata'):
                timestamp = timestamp.tz_convert('Asia/Kolkata')
            
            # Check session timing
            can_enter = self.can_enter_new_position(timestamp)
            
            # Check signal conditions
            should_enter = self.should_enter_long(row, timestamp)
            
            result = can_enter and should_enter
            
            # Debug logging for first few calls
            if hasattr(self, '_debug_call_count'):
                self._debug_call_count += 1
            else:
                self._debug_call_count = 1
                
            if self._debug_call_count <= 10:
                logger.info(f"can_open_long called #{self._debug_call_count}: "
                           f"can_enter={can_enter}, should_enter={should_enter}, result={result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in can_open_long: {e}")
            return False

    def open_long(self, row: pd.Series, current_time: datetime, position_manager) -> Optional[str]:
        """PRODUCTION INTERFACE: Position opening."""
        try:
            logger.info(f"Attempting to open long position at {current_time}")
            
            entry_price = row['close']
            
            # Use strategy parameters or defaults
            symbol = getattr(self, 'symbol', 'NIFTY')
            lot_size = self.config.get('lot_size', 1)
            tick_size = self.config.get('tick_size', 0.05)
            
            position_id = position_manager.open_position(
                symbol=symbol,
                entry_price=entry_price,
                timestamp=current_time,
                lot_size=lot_size,
                tick_size=tick_size
            )
            
            if position_id:
                # Update strategy state
                if hasattr(self, 'daily_stats'):
                    self.daily_stats['trades_today'] += 1
                self.last_signal_time = current_time
                
                logger.info(f"✅ Position opened: {position_id} @ {entry_price}")
                return position_id
            else:
                logger.warning("❌ Position manager returned None")
                return None
                
        except Exception as e:
            logger.error(f"Error in open_long: {e}")
            return None
    
    def should_exit(self, row, timestamp, position_manager):
        """Check if we should close position"""
        # Ensure timezone-aware timestamp
        timestamp = ensure_tz_aware(timestamp)
        
        # Always exit at session end
        if self.should_exit_for_session(timestamp):
            return True
        
        # Let position manager handle stop loss, take profit, and trailing stops
        return False
    
    def on_new_bar(self, row: pd.Series, current_time: datetime):
        """
        Called when a new bar is processed.
        
        Args:
            row: New bar data
            current_time: Bar timestamp
        """
        self.bars_processed += 1
        
        # Reset daily stats if new day
        if (self.daily_stats['session_start_time'] is None or 
            current_time.date() != self.daily_stats['session_start_time'].date()):
            
            self.daily_stats = {
                'trades_today': 0,
                'pnl_today': 0.0,
                'last_trade_time': None,
                'session_start_time': current_time
            }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get strategy information (for backtest compatibility).
        
        Returns:
            Strategy information dictionary
        """
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
                'atr': self.use_atr
            },
            'parameters': {
                'fast_ema': self.fast_ema,
                'slow_ema': self.slow_ema,
                'htf_period': self.htf_period,
                'base_sl_points': self.base_sl_points,
                'risk_per_trade_percent': self.risk_per_trade_percent
            },
            'constraints': {
                'long_only': True,
                'intraday_only': True,
                'max_trades_per_day': self.max_positions_per_day
            },
            'session': {
                'start': self.intraday_start.strftime('%H:%M'),
                'end': self.intraday_end.strftime('%H:%M'),
                'exit_before_close': self.exit_before_close
            },
            'daily_stats': self.daily_stats.copy()
        }
    
    def validate_parameters(self) -> List[str]:
        """
        Validate strategy parameters.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate EMA parameters
        if self.use_ema_crossover:
            if self.fast_ema >= self.slow_ema:
                errors.append("Fast EMA must be less than slow EMA")
            if self.fast_ema <= 0 or self.slow_ema <= 0:
                errors.append("EMA periods must be positive")
        
        # Validate HTF parameters
        if self.use_htf_trend:
            if self.htf_period <= 0:
                errors.append("HTF period must be positive")
        
        # Validate risk parameters
        if self.risk_per_trade_percent <= 0:
            errors.append("Risk per trade must be positive")
        if self.base_sl_points <= 0:
            errors.append("Base stop loss must be positive")
        
        # Validate session parameters
        if self.intraday_start >= self.intraday_end:
            errors.append("Session start must be before session end")
        
        return errors
    
    def reset(self):
        """Reset strategy state."""
        self.is_initialized = False
        self.current_position = None
        self.last_signal_time = None
        self.bars_processed = 0
        self.daily_stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'last_trade_time': None,
            'session_start_time': None
        }
        logger.info("Strategy reset to initial state")
    
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

    def is_session_live(self, current_time: datetime) -> bool:
        """Check if current time is within trading session."""
        # Ensure timezone-aware datetime
        current_time = ensure_tz_aware(current_time)
    
        # Extract time components for consistent comparison
        hour, minute = current_time.hour, current_time.minute
        current_minutes = hour * 60 + minute
    
        # Get session boundaries in minutes
        start_hour = self.session_params.get('intraday_start_hour', 9)
        start_min = self.session_params.get('intraday_start_min', 15)
        start_minutes = start_hour * 60 + start_min
    
        end_hour = self.session_params.get('intraday_end_hour', 15)
        end_min = self.session_params.get('intraday_end_min', 15)
        end_minutes = end_hour * 60 + end_min
    
        # Simple minutes-based comparison that ignores timezone
        return start_minutes <= current_minutes <= end_minutes

    def entry_signal(self, row: pd.Series) -> bool:
        # Collect signal conditions from enabled indicators only
        signal_conditions = []
        signal_reasons = []

        # EMA Crossover
        if self.config.get('use_ema_crossover', False):
            if ('fast_ema' in row and 'slow_ema' in row and
                not pd.isna(row['fast_ema']) and not pd.isna(row['slow_ema'])):
                # Check EMA crossover 
                fast_ema = row['fast_ema']
                slow_ema = row['slow_ema']
                if fast_ema > slow_ema:
                    signal_conditions.append(True)
                    signal_reasons.append(f"EMA Cross: {fast_ema:.2f} > {slow_ema:.2f}")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"EMA Cross: Fast EMA not above Slow EMA")
            else:
                signal_conditions.append(False)
                signal_reasons.append("EMA Cross: Data not available")

        # VWAP
        if self.config.get('use_vwap', False):
            if 'vwap' in row and not pd.isna(row['vwap']):
                if row['close'] > row['vwap']:
                    signal_conditions.append(True)
                    signal_reasons.append(f"VWAP: Price {row['close']:.2f} > VWAP {row['vwap']:.2f}")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"VWAP: Price {row['close']:.2f} not above VWAP {row['vwap']:.2f}")
            else:
                signal_conditions.append(False)
                signal_reasons.append("VWAP: Data not available")

        # MACD
        if self.config.get('use_macd', False):
            if all(x in row and not pd.isna(row[x]) for x in ['macd', 'macd_signal']):
                macd_val = row['macd']
                macd_signal = row['macd_signal']
                if macd_val > macd_signal:
                    signal_conditions.append(True)
                    signal_reasons.append(f"MACD: {macd_val:.2f} > Signal {macd_signal:.2f}")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"MACD: Not above signal line")
            else:
                signal_conditions.append(False)
                signal_reasons.append("MACD: Data not available")
        
        # Higher Timeframe Trend
        if self.config.get('use_htf_trend', False):
            if 'htf_trend' in row and not pd.isna(row['htf_trend']):
                if row['htf_trend'] > 0:  # Positive trend
                    signal_conditions.append(True)
                    signal_reasons.append(f"HTF Trend: Bullish ({row['htf_trend']:.2f})")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"HTF Trend: Not bullish")
            else:
                signal_conditions.append(False)
                signal_reasons.append("HTF Trend: Data not available")
                
        # RSI
        if self.config.get('use_rsi_filter', False):
            if 'rsi' in row and not pd.isna(row['rsi']):
                rsi_val = row['rsi']
                rsi_lower = self.config.get('rsi_lower', 30)
                rsi_upper = self.config.get('rsi_upper', 70)
                if rsi_lower < rsi_val < rsi_upper:
                    signal_conditions.append(True)
                    signal_reasons.append(f"RSI: {rsi_val:.2f} in range ({rsi_lower}-{rsi_upper})")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"RSI: {rsi_val:.2f} out of range")
            else:
                signal_conditions.append(False)
                signal_reasons.append("RSI: Data not available")
                
        # Bollinger Bands
        if self.config.get('use_bb', False):
            if all(x in row and not pd.isna(row[x]) for x in ['bb_upper', 'bb_lower']):
                price = row['close']
                if row['bb_lower'] < price < row['bb_upper']:
                    signal_conditions.append(True)
                    signal_reasons.append(f"BB: Price {price:.2f} within bands")
                else:
                    signal_conditions.append(False)
                    signal_reasons.append(f"BB: Price outside bands")
            else:
                signal_conditions.append(False)
                signal_reasons.append("BB: Data not available")

        # Store signal reasons for logging/debugging
        self.last_signal_reasons = signal_reasons
        
        # Must have at least one enabled indicator with valid signal
        if not signal_conditions:
            return False
            
        # All enabled indicators must agree (pass their conditions)
        return all(signal_conditions)

# For backwards compatibility, create an alias
MultiIndicatorStrategy = ModularIntradayStrategy

# Example usage
if __name__ == "__main__":
    # Test configuration
    test_params = {
        'use_ema_crossover': True,
        'fast_ema': 9,
        'slow_ema': 21,
        'use_macd': True,
        'use_vwap': True,
        'use_htf_trend': True,  # Now optional!
        'htf_period': 20,
        'use_rsi_filter': False,
        'base_sl_points': 15,
        'risk_per_trade_percent': 1.0,
        'session': {
            'is_intraday': True,
            'intraday_start_hour': 9,
            'intraday_start_min': 15,
            'intraday_end_hour': 15,
            'intraday_end_min': 15,
            'exit_before_close': 20
        },
        'max_trades_per_day': 10,
        'symbol': 'NIFTY24DECFUT',
        'lot_size': 15,
        'tick_size': 0.05
    }
    
    # Create strategy
    strategy = ModularIntradayStrategy(test_params)
    
    # Validate parameters
    errors = strategy.validate_parameters()
    if errors:
        print("Parameter validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ All parameters valid")
    
    # Show strategy info
    info = strategy.get_strategy_info()
    print(f"\n📊 Strategy: {info['name']}")
    print(f"🎯 Type: {info['type']}")
    print(f"📈 Indicators: {list(info['indicators_enabled'].keys())}")
    print(f"⏰ Session: {info['session']['start']} - {info['session']['end']}")
    
    print("\n✅ Strategy test completed successfully!")



"""
CONFIGURATION PARAMETER NAMING CONVENTION:
- Constructor parameter: __init__(self, config: Dict[str, Any], ...)
- Internal storage: self.config = config
- All parameter access: self.config.get('parameter_name', default)
- Session parameters: self.session_params = config.get('session', {})

MEMORY OPTIMIZATION:
- calculate_indicators() method supports memory_optimized=True for large datasets
- Uses self.config for parameter access, not self.params

CRITICAL CONSISTENCY REQUIREMENTS:
- Always use self.config.get() for parameter access
- Never use self.params (removed in favor of self.config)
- Session parameter access should use self.session_params.get()
- Indicator"""
