"""
core/indicators.py
Unified, parameter-driven indicator library for both backtest and live trading bot.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

def safe_divide(numerator, denominator, default=0.0):
    """Enhanced safe division with comprehensive error handling."""
    try:
        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return default
        return numerator / denominator
    except Exception:
        return default

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = calculate_ema(series, fast)
    slow_ema = calculate_ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': histogram})

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cum_vol = volume.cumsum()
    cum_tpv = (typical_price * volume).cumsum()
    return cum_tpv / cum_vol

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_htf_trend(close: pd.Series, period: int) -> pd.Series:
    return calculate_ema(close, period)

def calculate_stochastic(high, low, close, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = series.rolling(window=period).mean()
    sd = series.rolling(window=period).std()
    upper = ma + (sd * std)
    lower = ma - (sd * std)
    return upper, ma, lower

def calculate_ema_crossover_signals(fast_ema: pd.Series, slow_ema: pd.Series) -> pd.DataFrame:
    crossover = (fast_ema > slow_ema).fillna(False)
    # Set pandas option to eliminate warning
    pd.set_option('future.no_silent_downcasting', True)
    prev = crossover.shift(1).fillna(False)
    return pd.DataFrame({
        'bullish_cross': crossover & (~prev),
        'bearish_cross': (~crossover) & prev,
        'ema_bullish': crossover
    })

def calculate_macd_signals(macd_df: pd.DataFrame) -> pd.DataFrame:
    above = (macd_df['macd'] > macd_df['signal']).fillna(False)
    prev = above.shift().fillna(False)
    return pd.DataFrame({
        'macd_buy_signal': above & (~prev),
        'macd_sell_signal': (~above) & prev,
        'macd_bullish': above,
        'macd_histogram_positive': (macd_df['histogram'] > 0).fillna(False)
    })

def calculate_htf_signals(close: pd.Series, htf_ema: pd.Series) -> pd.DataFrame:
    bullish = (close > htf_ema).fillna(False)
    return pd.DataFrame({
        'htf_bullish': bullish,
        'htf_bearish': ~bullish
    })

def calculate_vwap_signals(close: pd.Series, vwap: pd.Series) -> pd.DataFrame:
    bullish = (close > vwap).fillna(False)
    return pd.DataFrame({
        'vwap_bullish': bullish,
        'vwap_bearish': ~bullish
    })

def calculate_rsi_signals(rsi: pd.Series, overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
    return pd.DataFrame({
        'rsi_oversold': (rsi <= oversold).fillna(False),
        'rsi_overbought': (rsi >= overbought).fillna(False),
        'rsi_neutral': ((rsi > oversold) & (rsi < overbought)).fillna(False)
    })

def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -> pd.DataFrame:
    """Calculate all technical indicators based on configuration params"""
    df = df.copy()

    # Log the enabled indicators for debugging
    enabled_indicators = [key for key in params if key.startswith('use_') and params.get(key)]
    logger.info(f"Calculating indicators: {enabled_indicators}")
     
    if not enabled_indicators:
        logger.warning("No indicators enabled in configuration")
        return df
     
    # === CENTRALIZED DATA VALIDATION ===
    logger.info(f"Validating data quality for {len(df)} rows with enabled indicators: {enabled_indicators}")

    # 1. Check for required columns based on ENABLED indicators only
    required_cols = ['close']
    if any(params.get(ind) for ind in ["use_atr", "use_stochastic", "use_bollinger_bands"]):
        required_cols.extend(['high', 'low'])
    if params.get("use_vwap"):
        required_cols.extend(['high', 'low', 'volume'])
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns for enabled indicators: {missing_cols}")
        # Don't calculate indicators that require missing columns
         
    # 2. Handle missing values in required columns
    if df[required_cols].isnull().any().any():
        null_counts = df[required_cols].isnull().sum()
        logger.warning(f"Fixing missing values in columns: {null_counts[null_counts > 0].to_dict()}")
        df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill')
    
    # 3. Fix negative/zero close prices which would cause calculation errors (needed for all indicators)
    if (df['close'] <= 0).any():
        neg_count = (df['close'] <= 0).sum()
        logger.warning(f"Fixing {neg_count} negative/zero prices")
        median_price = df['close'].median()
        if median_price <= 0:
            median_price = 1.0  # Fallback if median is also invalid
        df.loc[df['close'] <= 0, 'close'] = median_price
    
    # 4. Fix negative volume if present and volume indicators are used
    if (params.get("use_vwap") or params.get("use_volume_ma")) and 'volume' in df.columns and (df['volume'] < 0).any():
        # Only fix volume if volume-based indicators are enabled
        logger.warning(f"Fixing {(df['volume'] < 0).sum()} negative volume values")
        df.loc[df['volume'] < 0, 'volume'] = 0
        
    # For large datasets, process in chunks
    if len(df) > 5000:
        # Use efficient calculation approaches for large datasets
        logger.info(f"Using memory-optimized calculations for {len(df)} rows")
        
        # === EMA CROSSOVER ===
        if params.get("use_ema_crossover"):
            try:
                logger.info(f"Calculating EMA crossover with fast={params.get('fast_ema', 9)}, slow={params.get('slow_ema', 21)}")
                df['fast_ema'] = df['close'].ewm(
                    span=params.get('fast_ema', 9),
                    min_periods=params.get('fast_ema', 9)//2,
                    adjust=False
                ).mean()
                df['slow_ema'] = df['close'].ewm(
                    span=params.get('slow_ema', 21),
                    min_periods=params.get('slow_ema', 21)//2,
                    adjust=False
                ).mean()
                
                # Add crossover signals
                emacross = calculate_ema_crossover_signals(
                    df['fast_ema'], 
                    df['slow_ema']
                )
                df = df.join(emacross)
                logger.info(f"EMA crossover calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating EMA crossover: {str(e)}")
        
        # === MACD ===
        if params.get("use_macd"):
            try:
                logger.info(f"Calculating MACD with fast={params.get('macd_fast', 12)}, slow={params.get('macd_slow', 26)}, signal={params.get('macd_signal', 9)}")
                
                fast_ema = df['close'].ewm(span=params.get("macd_fast", 12), adjust=False).mean()
                slow_ema = df['close'].ewm(span=params.get("macd_slow", 26), adjust=False).mean()
                macd_line = fast_ema - slow_ema
                signal_line = macd_line.ewm(span=params.get("macd_signal", 9), adjust=False).mean()
                histogram = macd_line - signal_line
                
                df['macd'] = macd_line
                df['macd_signal'] = signal_line
                df['histogram'] = histogram
                
                # Add MACD signals
                macd_signals = calculate_macd_signals(pd.DataFrame({
                    'macd': macd_line, 
                    'signal': signal_line, 
                    'histogram': histogram
                }))
                df = df.join(macd_signals)
                
                logger.info(f"MACD calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating MACD: {str(e)}")
        
        # === VWAP ===
        if params.get("use_vwap"):
            required_vwap_cols = ['high', 'low', 'volume']
            if all(col in df.columns for col in required_vwap_cols):
                try:
                    logger.info(f"Calculating VWAP")
                    df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
                    # Add VWAP signals
                    df = df.join(calculate_vwap_signals(df['close'], df['vwap']))
                    logger.info(f"VWAP calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating VWAP: {str(e)}")
            else:
                missing = [col for col in required_vwap_cols if col not in df.columns]
                logger.error(f"Cannot calculate VWAP. Missing columns: {missing}")
        
        # === RSI FILTER ===
        if params.get("use_rsi_filter"):
            try:
                logger.info(f"Calculating RSI with length={params.get('rsi_length', 14)}")
                
                # Memory-efficient RSI calculation
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).ewm(span=params.get("rsi_length", 14), adjust=False).mean()
                loss = -delta.where(delta < 0, 0).ewm(span=params.get("rsi_length", 14), adjust=False).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                # Add RSI signals
                df = df.join(calculate_rsi_signals(
                    df['rsi'], 
                    params.get("rsi_overbought", 70), 
                    params.get("rsi_oversold", 30)
                ))
                
                logger.info(f"RSI calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating RSI: {str(e)}")
        
        # === HTF TREND ===
        if params.get("use_htf_trend"):
            try:
                logger.info(f"Calculating HTF trend with period={params.get('htf_period', 20)}")
                df['htf_ema'] = df['close'].ewm(
                    span=params.get("htf_period", 20), 
                    adjust=False
                ).mean()
                # Add HTF trend signals
                df = df.join(calculate_htf_signals(df['close'], df['htf_ema']))
                
                logger.info(f"HTF trend calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating HTF trend: {str(e)}")
        
        # === BOLLINGER BANDS ===
        if params.get("use_bollinger_bands"):
            try:
                logger.info(f"Calculating Bollinger Bands with period={params.get('bb_period', 20)}, std={params.get('bb_std', 2.0)}")
                period = params.get("bb_period", 20)
                std_dev = params.get("bb_std", 2.0)
                ma = df['close'].rolling(window=period).mean()
                sd = df['close'].rolling(window=period).std()
                df["bb_upper"] = ma + (sd * std_dev)
                df["bb_middle"] = ma
                df["bb_lower"] = ma - (sd * std_dev)
                
                logger.info(f"Bollinger Bands calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        
        # === STOCHASTIC ===
        if params.get("use_stochastic"):
            try:
                logger.info(f"Calculating Stochastic with k={params.get('stoch_k', 14)}, d={params.get('stoch_d', 3)}")
                k_period = params.get("stoch_k", 14)
                d_period = params.get("stoch_d", 3)
                lowest_low = df['low'].rolling(window=k_period).min()
                highest_high = df['high'].rolling(window=k_period).max()
                k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
                d = k.rolling(window=d_period).mean()
                df["stoch_k"] = k
                df["stoch_d"] = d
                
                logger.info(f"Stochastic calculated successfully")
            except Exception as e:
                logger.error(f"Error calculating Stochastic: {str(e)}")
        
        # === ATR ===
        if params.get("use_atr"):
            required_atr_cols = ['high', 'low']
            if all(col in df.columns for col in required_atr_cols):
                try:
                    logger.info(f"Calculating ATR with length={params.get('atr_len', 14)}")
                    df["atr"] = calculate_atr(
                        df["high"], 
                        df["low"], 
                        df["close"], 
                        params.get("atr_len", 14)
                    )
                    
                    logger.info(f"ATR calculated successfully")
                except Exception as e:
                    logger.error(f"Error calculating ATR: {str(e)}")
            else:
                missing = [col for col in required_atr_cols if col not in df.columns]
                logger.error(f"Cannot calculate ATR. Missing columns: {missing}")
        
        # === SIMPLE MOVING AVERAGES ===
        if params.get("use_ma"):
            logger.info(f"Calculating MAs with short={params.get('ma_short', 20)}, long={params.get('ma_long', 50)}")
            df["ma_short"] = df["close"].rolling(window=params.get("ma_short", 20)).mean()
            df["ma_long"] = df["close"].rolling(window=params.get("ma_long", 50)).mean()
        
        # === VOLUME MA ===
        if params.get("use_volume_ma"):
            if "volume" in df.columns:
                logger.info(f"Calculating volume MA with period={params.get('volume_ma_period', 20)}")
                df["volume_ma"] = df["volume"].rolling(window=params.get("volume_ma_period", 20)).mean()
                df["volume_ratio"] = safe_divide(df["volume"], df["volume_ma"])
            else:
                logger.error("Cannot calculate Volume MA. Missing 'volume' column")
    
    else:
        # Original calculation methods for smaller datasets
        if params.get("use_ema_crossover"):
            df['fast_ema'] = calculate_ema(df['close'], params.get("fast_ema", 9))
            df['slow_ema'] = calculate_ema(df['close'], params.get("slow_ema", 21))
            emacross = calculate_ema_crossover_signals(df['fast_ema'], df['slow_ema'])
            df = df.join(emacross)
            logger.info(f"EMA crossover calculated (regular)")
        
        if params.get("use_macd"):
            macd_df = calculate_macd(df['close'], params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
            df = df.join(macd_df)
            macd_signals = calculate_macd_signals(macd_df)
            df = df.join(macd_signals)
            logger.info(f"MACD calculated (regular)")
        
        if params.get("use_rsi_filter"):
            df['rsi'] = calculate_rsi(df['close'], params.get("rsi_length", 14))
            df = df.join(calculate_rsi_signals(df['rsi'], params.get("rsi_overbought", 70), params.get("rsi_oversold", 30)))
            logger.info(f"RSI calculated (regular)")
        
        if params.get("use_vwap"):
            required_vwap_cols = ['high', 'low', 'volume']
            if all(col in df.columns for col in required_vwap_cols):
                try:
                    df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
                    df = df.join(calculate_vwap_signals(df['close'], df['vwap']))
                    logger.info(f"VWAP calculated (regular)")
                except Exception as e:
                    logger.error(f"Error calculating VWAP: {str(e)}")
            else:
                missing = [col for col in required_vwap_cols if col not in df.columns]
                logger.error(f"Cannot calculate VWAP. Missing columns: {missing}")
        
        if params.get("use_htf_trend"):
            df['htf_ema'] = calculate_htf_trend(df['close'], params.get("htf_period", 20))
            df = df.join(calculate_htf_signals(df['close'], df['htf_ema']))
            logger.info(f"HTF trend calculated (regular)")
        
        if params.get("use_bollinger_bands"):
            upper, mid, lower = calculate_bollinger_bands(df['close'], params.get("bb_period", 20), params.get("bb_std", 2))
            df["bb_upper"], df["bb_middle"], df["bb_lower"] = upper, mid, lower
            logger.info(f"Bollinger Bands calculated (regular)")
        
        if params.get("use_stochastic"):
            k, d = calculate_stochastic(df['high'], df['low'], df['close'], params.get("stoch_k", 14), params.get("stoch_d", 3))
            df["stoch_k"], df["stoch_d"] = k, d
            logger.info(f"Stochastic calculated (regular)")
        
        if params.get("use_ma"):
            df["ma_short"] = calculate_sma(df["close"], params.get("ma_short", 20))
            df["ma_long"] = calculate_sma(df["close"], params.get("ma_long", 50))
            logger.info(f"MAs calculated (regular)")
        
        if params.get("use_atr"):
            required_atr_cols = ['high', 'low']
            if all(col in df.columns for col in required_atr_cols):
                df["atr"] = calculate_atr(df["high"], df["low"], df["close"], params.get("atr_len", 14))
                logger.info(f"ATR calculated (regular)")
            else:
                missing = [col for col in required_atr_cols if col not in df.columns]
                logger.error(f"Cannot calculate ATR. Missing columns: {missing}")
        
        if params.get("use_volume_ma"):
            if "volume" in df.columns:
                df["volume_ma"] = calculate_sma(df["volume"], params.get("volume_ma_period", 20))
                df["volume_ratio"] = safe_divide(df["volume"], df["volume_ma"])
                logger.info(f"Volume MA calculated (regular)")
            else:
                logger.error("Cannot calculate Volume MA. Missing 'volume' column")
    
    # Log the total indicators calculated
    calculated_indicators = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'datetime']]
    logger.info(f"Calculated {len(calculated_indicators)} indicators: {calculated_indicators[:5]}...")

    return df

from typing import Tuple

# --- Incremental EMA ---
def update_ema(price: float, prev_ema: float, period: int) -> float:
    """
    Incremental EMA update formula.
    """
    alpha = 2 / (period + 1)
    return (price - prev_ema) * alpha + prev_ema

class IncrementalEMA:
    """
    Incremental EMA tracker holding its own state.
    """
    def __init__(self, period: int, first_price: float = None):
        self.period = period
        self.ema = first_price
    def update(self, price: float) -> float:
        if self.ema is None:
            self.ema = price
        else:
            self.ema = update_ema(price, self.ema, self.period)
        return self.ema

# --- Incremental MACD as previously integrated ---
class IncrementalMACD:
    """
    Incremental MACD, Signal line, Histogram.
    """
    def __init__(self, fast=12, slow=26, signal=9, first_price=None):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.fast_ema = first_price
        self.slow_ema = first_price
        self.macd = 0.0
        self.signal_line = 0.0
    def update(self, price: float) -> Tuple[float, float, float]:
        if self.fast_ema is None or self.slow_ema is None:
            self.fast_ema = self.slow_ema = price
            self.macd = 0.0
            self.signal_line = 0.0
        else:
            self.fast_ema = update_ema(price, self.fast_ema, self.fast)
            self.slow_ema = update_ema(price, self.slow_ema, self.slow)
            self.macd = self.fast_ema - self.slow_ema
            self.signal_line = update_ema(self.macd, self.signal_line, self.signal)
        histogram = self.macd - self.signal_line
        return self.macd, self.signal_line, histogram

# --- Incremental VWAP (per session/day) ---
class IncrementalVWAP:
    """
    Incremental VWAP for intraday/session use with robust error handling.
    """
    def __init__(self):
        self.cum_tpv = 0.0
        self.cum_vol = 0.0
        self.last_vwap = None  # Store last valid VWAP for fallback
        
    def update(self, price, volume, high=None, low=None, close=None):
        try:
            # Validate inputs
            if pd.isna(price) or price <= 0:
                return self.last_vwap if self.last_vwap is not None else price
                
            # Handle invalid volume
            if pd.isna(volume) or volume < 0:
                volume = 0
                
            if high is not None and low is not None and close is not None:
                if not (pd.isna(high) or pd.isna(low) or pd.isna(close)):
                    typical_price = (high + low + close) / 3
                else:
                    typical_price = price
            else:
                typical_price = price
                
            self.cum_tpv += typical_price * volume
            self.cum_vol += volume
            
            # Safe calculation with fallback
            if self.cum_vol > 0:
                self.last_vwap = self.cum_tpv / self.cum_vol
            else:
                self.last_vwap = typical_price
                
            return self.last_vwap
        except Exception as e:
            logger.error(f"VWAP calculation error: {str(e)}")
            return self.last_vwap if self.last_vwap is not None else price
            
    def reset(self):
        self.cum_tpv = 0.0
        self.cum_vol = 0.0
        # Don't reset last_vwap to maintain a value during transitions

class IncrementalATR:
    """
    Incremental ATR, using Welles Wilder smoothing with robust error handling.
    """
    def __init__(self, period=14, first_close=None):
        self.period = period
        self.atr = None
        self.prev_close = first_close
        self.initialized = False
        self.tr_queue = []
        
    def update(self, high, low, close):
        try:
            # Validate inputs
            if pd.isna(high) or pd.isna(low) or pd.isna(close):
                return self.atr if self.atr is not None else 0.0
                
            # Ensure high >= low (data consistency)
            if high < low:
                high, low = low, high  # Swap if inverted
                
            if self.prev_close is None:
                tr = high - low
            else:
                tr = max(
                    high - low,
                    abs(high - self.prev_close),
                    abs(low - self.prev_close)
                )
                
            if not self.initialized:
                self.tr_queue.append(tr)
                if len(self.tr_queue) == self.period:
                    self.atr = sum(self.tr_queue) / self.period
                    self.initialized = True
            else:
                self.atr = (self.atr * (self.period - 1) + tr) / self.period
                
            self.prev_close = close
            return self.atr if self.atr is not None else tr
        except Exception as e:
            logger.error(f"ATR calculation error: {str(e)}")
            return self.atr if self.atr is not None else 0.0

class IncrementalEMA:
    """
    Incremental EMA tracker with robust error handling.
    """
    def __init__(self, period: int, first_price: float = None):
        self.period = period
        self.ema = first_price
        
    def update(self, price: float) -> float:
        try:
            # Validate input
            if pd.isna(price):
                return self.ema  # Return last value if available
                
            if self.ema is None:
                self.ema = price
            else:
                self.ema = update_ema(price, self.ema, self.period)
            return self.ema
        except Exception as e:
            logger.error(f"EMA calculation error: {str(e)}")
            return self.ema if self.ema is not None else price

class IncrementalMACD:
    """
    Incremental MACD with robust error handling.
    """
    def __init__(self, fast=12, slow=26, signal=9, first_price=None):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.fast_ema = first_price
        self.slow_ema = first_price
        self.macd = 0.0
        self.signal_line = 0.0
        
    def update(self, price: float) -> Tuple[float, float, float]:
        try:
            # Validate input
            if pd.isna(price):
                return self.macd, self.signal_line, (self.macd - self.signal_line)
                
            if self.fast_ema is None or self.slow_ema is None:
                self.fast_ema = self.slow_ema = price
                self.macd = 0.0
                self.signal_line = 0.0
            else:
                self.fast_ema = update_ema(price, self.fast_ema, self.fast)
                self.slow_ema = update_ema(price, self.slow_ema, self.slow)
                self.macd = self.fast_ema - self.slow_ema
                self.signal_line = update_ema(self.macd, self.signal_line, self.signal)
                
            histogram = self.macd - self.signal_line
            return self.macd, self.signal_line, histogram
        except Exception as e:
            logger.error(f"MACD calculation error: {str(e)}")
            return self.macd, self.signal_line, (self.macd - self.signal_line)


"""
PARAMETER NAMING CONVENTION:
- Main function: calculate_all_indicators(df: pd.DataFrame, params: Dict)
- Parameter name 'params' is MANDATORY for interface compatibility
- All internal usage: params.get('parameter_name', default)

INTERFACE REQUIREMENT:
- The 'params' parameter name CANNOT be changed as it's used by:
  * researchStrategy.py: calculate_all_indicators(df, self.config)  
  * liveStrategy.py: calculate_all_indicators(df, self.config)
  * Multiple indicator calculation functions

CRITICAL: 
- Strategy modules pass their self.config as 'params' to this module
- This creates the interface boundary between 'config' (strategies) and 'params' (indicators)
- Do NOT change 'params' parameter name without updating ALL calling code
"""
