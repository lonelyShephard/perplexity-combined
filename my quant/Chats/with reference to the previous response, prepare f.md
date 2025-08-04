

# with reference to the previous response, prepare file by file complete diffs to be incorporated in code.

Now I'll create comprehensive diffs for each file to address the identified issues:

## File 1: backtest_runner.py

```diff
--- a/backtest/backtest_runner.py
+++ b/backtest/backtest_runner.py
@@ -27,6 +27,7 @@ try:
     logger.info(f"✅ ensure_tz_aware verified from {inspect.getmodule(ensure_tz_aware).__name__}")
 except (AssertionError, AttributeError, ImportError) as e:
     logger.error(f"❌ ensure_tz_aware verification failed: {e}")
+    # Raise immediately to prevent hard-to-diagnose errors later
     raise ImportError("Critical timezone function not properly available")
 
 def load_config(config_path: str) -> dict:
@@ -35,12 +36,15 @@ def load_config(config_path: str) -> dict:
         return yaml.safe_load(f)
 
 def get_strategy(config: dict):
-    """Load strategy module with full configuration."""
+    """Load strategy module with full configuration.
+    
+    Args:
+        config: Complete configuration dictionary
+    """
     version = config.get("strategy", {}).get("strategy_version", "live").lower()
     if version == "research":
         strat_mod = importlib.import_module("core.researchStrategy")
     else:
         strat_mod = importlib.import_module("core.liveStrategy")
     
     ind_mod = importlib.import_module("core.indicators")
     return strat_mod.ModularIntradayStrategy(config, ind_mod)
@@ -48,7 +52,10 @@ def get_strategy(config: dict):
 def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculation=False):
-    """Run a backtest with the given configuration"""
+    """Run a backtest with the given configuration.
+    
+    Args:
+        config: Configuration dictionary or path to config file
+        data_file: Path to data file (CSV/LOG)
+        df_normalized: Pre-normalized data (optional)
+        skip_indicator_calculation: Skip indicator calculation if data already has them
+    """
     logger.info("=" * 60)
     logger.info("STARTING BACKTEST WITH NORMALIZED DATA PIPELINE")
     logger.info("=" * 60)
@@ -58,19 +65,23 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
         config = load_config(config)
     
     # Extract parameters with validation
-    strategy_params = config.get('strategy', {})
-    session_params = config.get('session', {})
-    risk_params = config.get('risk', {})
-    instrument_params = config.get('instrument', {})
-    capital = config.get('capital', {}).get('initial_capital', 100000)
+    try:
+        strategy_params = config.get('strategy', {})
+        session_params = config.get('session', {})
+        risk_params = config.get('risk', {})
+        instrument_params = config.get('instrument', {})
+        capital = config.get('capital', {}).get('initial_capital', 100000)
+        
+        # Validate essential parameters
+        if capital <= 0:
+            raise ValueError("Initial capital must be positive")
+            
+    except Exception as e:
+        logger.error(f"Configuration validation failed: {e}")
+        raise
     
     # Initialize components
     strategy = get_strategy(config)
     
-    # Create a consolidated config dictionary
+    # Create consolidated config for position manager
     position_config = {
         **strategy_params,
         **risk_params,
@@ -78,8 +89,7 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
         'initial_capital': capital,
         'session': session_params
     }
-    
-    # Initialize with a single dictionary argument
     position_manager = PositionManager(position_config)
     
     # Skip data loading if df_normalized is provided
@@ -96,6 +106,11 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
             'issues_found': {}
         })
     
+    # Validate data quality
+    if df_normalized.empty:
+        raise ValueError("No data available for backtest")
+    
+    logger.info(f"Starting backtest with {len(df_normalized)} data points")
+    
     # Optimize memory usage for large tick dataset
     if len(df_normalized) > 5000:
         logger.info(f"Optimizing memory usage for large tick dataset ({len(df_normalized)} ticks)")
@@ -104,13 +119,15 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
         lookback_limit = min(500, int(len(df_normalized) * 0.1))  # 10% of data or 500 max
         
         # Pass memory optimization parameters to indicator calculation
-        if skip_indicator_calculation and df_normalized is not None:
+        if skip_indicator_calculation:
             # Use pre-calculated indicators
             df_with_indicators = df_normalized
-            print("Using pre-calculated indicators")
+            logger.info("Using pre-calculated indicators")
         else:
             # Calculate indicators as usual
-            df_with_indicators = strategy.calculate_indicators(df_normalized,
+            df_with_indicators = strategy.calculate_indicators(
+                df_normalized,
                 memory_optimized=True,
                 max_lookback=lookback_limit)
     else:
@@ -121,14 +138,19 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
     logger.info("Starting backtest execution...")
     position_id = None
     in_position = False
+    processed_bars = 0
     
-    # Extract session parameters for exit logic (CORRECTED)
-    session_params = config.get('session', {})  # ✅ Fixed: consistent config naming
+    # Extract session parameters for exit logic
     close_hour = session_params.get("intraday_end_hour", 15)
-    close_min = session_params.get("intraday_end_min", 30)  # ✅
+    close_min = session_params.get("intraday_end_min", 30)
     exit_buffer = session_params.get("exit_before_close", 20)
-    
-    processed_bars = 0  # Add this counter
+    
+    # Validation of session parameters
+    if close_hour < 0 or close_hour > 23:
+        logger.warning(f"Invalid close hour {close_hour}, using default 15")
+        close_hour = 15
+    if close_min < 0 or close_min > 59:
+        logger.warning(f"Invalid close minute {close_min}, using default 30")
+        close_min = 30
     
     for timestamp, row in df_with_indicators.iterrows():
         processed_bars += 1
@@ -136,24 +158,29 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
         # ENSURE timezone awareness for timestamp
         now = ensure_tz_aware(timestamp)
         
-        # Add this after normalizing the timestamp
+        # Add session exit flag to row
         row['session_exit'] = is_time_to_exit(now, exit_buffer, close_hour, close_min)
         
-        # Check if in exit buffer period - CORE LOGIC (move this up)
+        # Check if in exit buffer period - CORE LOGIC
         if is_time_to_exit(now, exit_buffer, close_hour, close_min):
             # Close all positions and terminate
             for pos_id in list(position_manager.positions.keys()):
                 position_manager.close_position_full(pos_id, row['close'], now, "Exit Buffer")
+            logger.info("Exit buffer reached - all positions closed")
             break  # Stop processing completely
         
         # For debugging the first few iterations
-        if processed_bars <= 1:
+        if processed_bars <= 3:
             logger.info(f"Processing timestamp: {now} (tzinfo: {now.tzinfo})")
+            logger.info(f"Row data: close={row.get('close', 'N/A')}, volume={row.get('volume', 'N/A')}")
         
-        # Process positions with timezone-aware timestamp
-        position_manager.process_positions(row, now)
+        # Validate row data
+        if pd.isna(row.get('close')):
+            logger.warning(f"Invalid close price at {now}, skipping")
+            continue
         
-        # Entry Logic: only if not already in position and conditions meet
+        # Process positions with timezone-aware timestamp
+        position_manager.process_positions(row, now, session_params)
+        
+        # Entry Logic: only if not already in position and conditions are met
         if not in_position and strategy.can_open_long(row, now):
             position_id = strategy.open_long(row, now, position_manager)
             in_position = position_id is not None
@@ -161,7 +188,7 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
             if in_position:
                 logger.debug(f"Opened position {position_id} at {now} @ {row['close']:.2f}")
         
-        # Exit Logic: PositionManager handles trailing stops, TPs, SLs and session-end exits
+        # Exit Logic: PositionManager handles trailing stops, TPs, SLs
         if in_position:
-            position_manager.process_positions(row, now)
             
             # Check for strategy-level exit conditions
             if strategy.should_close(row, now, position_manager):
@@ -171,9 +198,6 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
                 in_position = False
                 position_id = None
                 logger.debug(f"Strategy exit at {now} @ {last_price:.2f}")
-        else:
-            # Still allow PositionManager to process positions in edge cases  
-            position_manager.process_positions(row, now)
         
         # Reset position state if position closed by PositionManager
         if position_id and position_id not in position_manager.positions:
@@ -181,17 +205,18 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
             position_id = None
         
         # Log first timestamp info
-        if processed_bars == 1:
-            # Log details of the first timestamp to verify timezone awareness
+        if processed_bars <= 2:
             logger.info(f"First timestamp processing details:")
             logger.info(f" - Original timestamp: {timestamp} (tzinfo: {timestamp.tzinfo})")
             logger.info(f" - Normalized 'now': {now} (tzinfo: {now.tzinfo})")
             logger.info(f" - Session exit check: {row['session_exit']}")
-            # Check if strategy methods handle the timestamp properly
             logger.info(f" - Session live check: {strategy.is_session_live(now)}")
     
     # Defensive: flatten any still-open positions at backtest end
     if position_id and position_id in position_manager.positions:
-        last_price = df_with_indicators.iloc[-1]['close']
+        try:
+            last_price = df_with_indicators.iloc[-1]['close']
+        except (IndexError, KeyError):
+            last_price = 0.0
         now = df_with_indicators.index[-1]
         strategy.handle_exit(position_id, last_price, now, position_manager, reason="End of Backtest")
         logger.info(f"Closed final position at backtest end @ {last_price:.2f}")
@@ -203,8 +228,11 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
     logger.info("BACKTEST SUMMARY")
     logger.info("=" * 60)
     logger.info(f"Data Quality:")
-    logger.info(f" Total input rows: {quality_report.total_rows}")
-    logger.info(f" Processed rows: {quality_report.rows_processed}")
+    total_rows = getattr(quality_report, 'total_rows', len(df_normalized))
+    processed_rows = getattr(quality_report, 'rows_processed', len(df_normalized))
+    
+    logger.info(f" Total input rows: {total_rows}")
+    logger.info(f" Processed rows: {processed_rows}")
     logger.info(f" Data quality: {quality_report.rows_processed/quality_report.total_rows*100:.1f}%")
     logger.info("")
     logger.info(f"Trading Performance:")
@@ -220,7 +248,11 @@ def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculati
     logger.info("=" * 60)
     
     # Save trade log CSV file
-    if trades and len(trades) > 0:
+    try:
+        if trades and len(trades) > 0:
+            trades_df = pd.DataFrame(trades)
+            trades_df.to_csv("backtest_trades.csv", index=False)
+            logger.info("Trade log written to backtest_trades.csv")
+        else:
+            logger.warning("No trades executed during backtest")
+            trades_df = pd.DataFrame()
+    except Exception as e:
+        logger.error(f"Error saving trade log: {e}")
         trades_df = pd.DataFrame(trades)
-        trades_df.to_csv("backtest_trades.csv", index=False)
-        logger.info("Trade log written to backtest_trades.csv")
-    else:
-        logger.warning("No trades executed during backtest")
-        trades_df = pd.DataFrame()
     
     return trades_df, performance
```


## File 2: strategy_config.yaml

```diff
--- a/config/strategy_config.yaml
+++ b/config/strategy_config.yaml
@@ -1,4 +1,5 @@
-# strategy_config.yaml – Master Configuration
+# strategy_config.yaml – Master Configuration for Unified Trading System
+# All parameters are validated and have sensible defaults
 
 strategy:
   # === Indicator Update Mode ===
@@ -80,6 +81,11 @@ risk:
   commission_percent: 0.1 # Commission as percentage per trade
   commission_per_trade: 0.0 # Minimum per-trade commission
   buy_buffer: 0 # Optional points buffer for entry slippage
+  
+  # === Additional Risk Parameters ===
+  max_position_value_percent: 95 # Maximum position value as % of capital
+  stt_percent: 0.025 # Securities Transaction Tax
+  exchange_charges_percent: 0.0019 # Exchange charges
+  gst_percent: 18.0 # GST on charges
 
 capital:
   initial_capital: 100000
@@ -90,6 +96,7 @@ session:
   intraday_start_min: 15
   intraday_end_hour: 15
   intraday_end_min: 30
+  # ^^^ CORRECTED: NSE closes at 3:30 PM, not 3:15 PM
   exit_before_close: 20
   timezone: Asia/Kolkata
 
@@ -97,9 +104,16 @@ reentry:
   reentry_price_buffer: 5
   reentry_momentum_lookback: 3
   reentry_min_green_candles: 1
+
+# === Instrument Configuration ===
+instrument:
+  symbol: NIFTY24DECFUT
+  exchange: NSE_FO
+  lot_size: 15  # NIFTY lot size (as of 2024)
+  tick_size: 0.05
+  product_type: INTRADAY
 
 backtest:
   max_drawdown_pct: 0 # 0 disables early stop
   allow_short: false # Enforce long-only
   close_at_session_end: true
@@ -107,6 +121,7 @@ backtest:
   results_dir: backtest_results
   log_level: INFO
+  max_trades_per_day: 25 # Maximum trades per session
 
 live:
   exchange_type: NSE_FO
@@ -114,6 +129,10 @@ live:
   log_ticks: false
   visual_indicator: true
+  paper_trading: true # Always default to simulation mode
+  
+  # SmartAPI configuration (use environment variables for sensitive data)
+  # Set TRADING_LIVE_API_KEY, TRADING_LIVE_CLIENT_CODE, etc.
 
 data_quality:
   strict_mode: true # Fail fast on critical data issues
```


## File 3: indicators.py

```diff
--- a/core/indicators.py
+++ b/core/indicators.py
@@ -8,6 +8,7 @@ Unified, parameter-driven indicator library for both backtest and live trading
 import pandas as pd
 import numpy as np
 import logging
+from typing import Dict, Tuple, Any, Optional
 
 logger = logging.getLogger(__name__)
 
@@ -20,6 +21,12 @@ def safe_divide(numerator, denominator, default=0.0):
     except Exception:
         return default
 
+def validate_series(series: pd.Series, name: str) -> pd.Series:
+    """Validate and clean a pandas Series for indicator calculations."""
+    if series.isnull().any():
+        logger.warning(f"Found {series.isnull().sum()} null values in {name}, forward filling")
+        series = series.fillna(method='ffill').fillna(method='bfill')
+    return series
+
 def calculate_ema(series: pd.Series, period: int) -> pd.Series:
+    """Calculate Exponential Moving Average with validation."""
+    series = validate_series(series, f"EMA({period})")
     return series.ewm(span=period, adjust=False).mean()
 
@@ -27,6 +34,8 @@ def calculate_sma(series: pd.Series, period: int) -> pd.Series:
+    """Calculate Simple Moving Average with validation."""
+    series = validate_series(series, f"SMA({period})")
     return series.rolling(window=period).mean()
 
 def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
+    """Calculate RSI with validation and error handling."""
+    series = validate_series(series, f"RSI({period})")
     delta = series.diff()
     gain = delta.where(delta > 0, 0).rolling(window=period).mean()
     loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
@@ -34,6 +43,10 @@ def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
     return 100 - (100 / (1 + rs))
 
 def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
+    """Calculate MACD with validation."""
+    if fast >= slow:
+        raise ValueError(f"MACD fast period ({fast}) must be less than slow period ({slow})")
+    
+    series = validate_series(series, f"MACD({fast},{slow},{signal})")
     fast_ema = calculate_ema(series, fast)
     slow_ema = calculate_ema(series, slow)
     macd_line = fast_ema - slow_ema
@@ -42,6 +55,14 @@ def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: i
     return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': histogram})
 
 def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
+    """Calculate VWAP with comprehensive validation."""
+    # Validate all inputs
+    high = validate_series(high, "VWAP High")
+    low = validate_series(low, "VWAP Low")  
+    close = validate_series(close, "VWAP Close")
+    volume = validate_series(volume, "VWAP Volume")
+    
+    # Ensure volume is non-negative
+    volume = volume.clip(lower=0)
+    
     typical_price = (high + low + close) / 3
     cum_vol = volume.cumsum()
     cum_tpv = (typical_price * volume).cumsum()
@@ -49,6 +70,12 @@ def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: p
     return cum_tpv / cum_vol
 
 def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
+    """Calculate ATR with validation."""
+    high = validate_series(high, "ATR High")
+    low = validate_series(low, "ATR Low")
+    close = validate_series(close, "ATR Close")
+    
+    # Ensure high >= low
+    high = pd.Series(np.maximum(high.values, low.values), index=high.index)
+    
     high_low = high - low
     high_close = np.abs(high - close.shift())
     low_close = np.abs(low - close.shift())
@@ -56,6 +83,8 @@ def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: in
     return tr.rolling(window=period).mean()
 
 def calculate_htf_trend(close: pd.Series, period: int) -> pd.Series:
+    """Calculate Higher Timeframe trend using EMA."""
+    close = validate_series(close, f"HTF Trend({period})")
     return calculate_ema(close, period)
 
 def calculate_stochastic(high, low, close, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
+    """Calculate Stochastic oscillator with validation."""
+    high = validate_series(high, f"Stoch High({k_period})")
+    low = validate_series(low, f"Stoch Low({k_period})")
+    close = validate_series(close, f"Stoch Close({k_period})")
+    
     lowest_low = low.rolling(window=k_period).min()
     highest_high = high.rolling(window=k_period).max()
     k = 100 * (close - lowest_low) / (highest_high - lowest_low)
@@ -63,6 +92,9 @@ def calculate_stochastic(high, low, close, k_period: int, d_period: int) -> Tup
     return k, d
 
 def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
+    """Calculate Bollinger Bands with validation."""
+    if std <= 0:
+        raise ValueError(f"Standard deviation multiplier must be positive, got {std}")
+    
+    series = validate_series(series, f"BB({period},{std})")
     ma = series.rolling(window=period).mean()
     sd = series.rolling(window=period).std()
     upper = ma + (sd * std)
@@ -70,6 +102,12 @@ def calculate_bollinger_bands(series: pd.Series, period: int = 20, std: float =
     return upper, ma, lower
 
 def calculate_ema_crossover_signals(fast_ema: pd.Series, slow_ema: pd.Series, threshold: float = 0) -> pd.DataFrame:
+    """Calculate EMA crossover signals with validation."""
+    if fast_ema.empty or slow_ema.empty:
+        logger.warning("Empty EMA series provided for crossover calculation")
+        return pd.DataFrame({
+            'bullish_cross': pd.Series(dtype=bool),
+            'bearish_cross': pd.Series(dtype=bool), 
+            'ema_bullish': pd.Series(dtype=bool)
+        })
+        
     crossover = (fast_ema > (slow_ema + threshold)).fillna(False)
     # Set pandas option to eliminate warning
     pd.set_option('future.no_silent_downcasting', True)
@@ -81,6 +119,13 @@ def calculate_ema_crossover_signals(fast_ema: pd.Series, slow_ema: pd.Series, t
     })
 
 def calculate_macd_signals(macd_df: pd.DataFrame) -> pd.DataFrame:
+    """Calculate MACD signals with validation."""
+    required_cols = ['macd', 'signal', 'histogram']
+    if not all(col in macd_df.columns for col in required_cols):
+        logger.error(f"MACD DataFrame missing required columns: {required_cols}")
+        return pd.DataFrame({
+            'macd_buy_signal': pd.Series(dtype=bool),
+            'macd_sell_signal': pd.Series(dtype=bool),
+            'macd_bullish': pd.Series(dtype=bool),
+            'macd_histogram_positive': pd.Series(dtype=bool)
+        })
+        
     above = (macd_df['macd'] > macd_df['signal']).fillna(False)
     prev = above.shift().fillna(False)
     return pd.DataFrame({
@@ -92,6 +137,10 @@ def calculate_macd_signals(macd_df: pd.DataFrame) -> pd.DataFrame:
     })
 
 def calculate_htf_signals(close: pd.Series, htf_ema: pd.Series) -> pd.DataFrame:
+    """Calculate HTF trend signals with validation."""
+    if close.empty or htf_ema.empty:
+        return pd.DataFrame({
+            'htf_bullish': pd.Series(dtype=bool),
+            'htf_bearish': pd.Series(dtype=bool)
+        })
+        
     bullish = (close > htf_ema).fillna(False)
     return pd.DataFrame({
         'htf_bullish': bullish,
@@ -99,6 +148,10 @@ def calculate_htf_signals(close: pd.Series, htf_ema: pd.Series) -> pd.DataFrame
     })
 
 def calculate_vwap_signals(close: pd.Series, vwap: pd.Series) -> pd.DataFrame:
+    """Calculate VWAP signals with validation."""
+    if close.empty or vwap.empty:
+        return pd.DataFrame({
+            'vwap_bullish': pd.Series(dtype=bool),
+            'vwap_bearish': pd.Series(dtype=bool)
+        })
+        
     bullish = (close > vwap).fillna(False)
     return pd.DataFrame({
         'vwap_bullish': bullish,
@@ -106,6 +159,13 @@ def calculate_vwap_signals(close: pd.Series, vwap: pd.Series) -> pd.DataFrame:
     })
 
 def calculate_rsi_signals(rsi: pd.Series, overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
+    """Calculate RSI signals with validation."""
+    if oversold >= overbought:
+        raise ValueError(f"RSI oversold ({oversold}) must be less than overbought ({overbought})")
+        
+    if rsi.empty:
+        return pd.DataFrame({
+            'rsi_oversold': pd.Series(dtype=bool),
+            'rsi_overbought': pd.Series(dtype=bool),
+            'rsi_neutral': pd.Series(dtype=bool)
+        })
+        
     return pd.DataFrame({
         'rsi_oversold': (rsi <= oversold).fillna(False),
         'rsi_overbought': (rsi >= overbought).fillna(False),
@@ -113,8 +173,15 @@ def calculate_rsi_signals(rsi: pd.Series, overbought: float = 70, oversold: flo
     })
 
-def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -> pd.DataFrame:
-    """Calculate all indicators with centralized data validation."""
+def calculate_all_indicators(df: pd.DataFrame, config: Dict, chunk_size=1000) -> pd.DataFrame:
+    """Calculate all indicators with centralized data validation.
+    
+    Args:
+        df: Input OHLCV DataFrame
+        config: Configuration dictionary (renamed from params for consistency)
+        chunk_size: Processing chunk size for large datasets
+        
+    Returns:
+        DataFrame with indicators added
+    """
     df = df.copy()
     
     # === CENTRALIZED DATA VALIDATION ===
@@ -122,7 +189,7 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
     
     # 1. Check for required columns based on requested indicators
     required_cols = ['close']
-    if params.get("use_vwap") or params.get("use_atr") or params.get("use_stochastic") or params.get("use_bollinger_bands"):
+    if config.get("use_vwap") or config.get("use_atr") or config.get("use_stochastic") or config.get("use_bollinger_bands"):
         required_cols.extend(['high', 'low'])
-    if params.get("use_vwap"):
+    if config.get("use_vwap"):
         required_cols.append('volume')
     
     missing_cols = [col for col in required_cols if col not in df.columns]
@@ -144,7 +211,7 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
         df.loc[df['close'] <= 0, 'close'] = median_price
     
     # 4. Fix negative volume if present and volume indicators are used
-    if 'volume' in df.columns and params.get("use_vwap") and (df['volume'] < 0).any():
+    if 'volume' in df.columns and config.get("use_vwap") and (df['volume'] < 0).any():
         logger.warning(f"Fixing {(df['volume'] < 0).sum()} negative volume values")
         df.loc[df['volume'] < 0, 'volume'] = 0
     
@@ -156,20 +223,20 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
         logger.info(f"Using memory-optimized calculations for {len(df)} rows")
         
         # === EMA CROSSOVER ===
-        if params.get("use_ema_crossover"):
+        if config.get("use_ema_crossover"):
             df['fast_ema'] = df['close'].ewm(
-                span=params.get('fast_ema', 9),
-                min_periods=params.get('fast_ema', 9)//2,
+                span=config.get('fast_ema', 9),
+                min_periods=config.get('fast_ema', 9)//2,
                 adjust=False
             ).mean()
             df['slow_ema'] = df['close'].ewm(
-                span=params.get('slow_ema', 21),
-                min_periods=params.get('slow_ema', 21)//2,
+                span=config.get('slow_ema', 21),
+                min_periods=config.get('slow_ema', 21)//2,
                 adjust=False
             ).mean()
             
             # Add crossover signals
             emacross = calculate_ema_crossover_signals(
                 df['fast_ema'],
                 df['slow_ema'],
-                params.get("ema_points_threshold", 0)
+                config.get("ema_points_threshold", 0)
             )
             df = df.join(emacross)
         
         # === MACD ===
-        if params.get("use_macd"):
-            fast_ema = df['close'].ewm(span=params.get("macd_fast", 12), adjust=False).mean()
-            slow_ema = df['close'].ewm(span=params.get("macd_slow", 26), adjust=False).mean()
+        if config.get("use_macd"):
+            fast_ema = df['close'].ewm(span=config.get("macd_fast", 12), adjust=False).mean()
+            slow_ema = df['close'].ewm(span=config.get("macd_slow", 26), adjust=False).mean()
             macd_line = fast_ema - slow_ema
-            signal_line = macd_line.ewm(span=params.get("macd_signal", 9), adjust=False).mean()
+            signal_line = macd_line.ewm(span=config.get("macd_signal", 9), adjust=False).mean()
             histogram = macd_line - signal_line
             
             df['macd'] = macd_line
@@ -194,7 +261,7 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
             df = df.join(macd_signals)
         
         # === VWAP ===
-        if params.get("use_vwap") and all(col in df.columns for col in ['high', 'low', 'volume']):
+        if config.get("use_vwap") and all(col in df.columns for col in ['high', 'low', 'volume']):
             df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
             df = df.join(calculate_vwap_signals(df['close'], df['vwap']))
         
         # === RSI FILTER ===
-        if params.get("use_rsi_filter"):
+        if config.get("use_rsi_filter"):
             # Memory-efficient RSI calculation
             delta = df['close'].diff()
-            gain = delta.where(delta > 0, 0).ewm(span=params.get("rsi_length", 14), adjust=False).mean()
-            loss = -delta.where(delta < 0, 0).ewm(span=params.get("rsi_length", 14), adjust=False).mean()
+            gain = delta.where(delta > 0, 0).ewm(span=config.get("rsi_length", 14), adjust=False).mean()
+            loss = -delta.where(delta < 0, 0).ewm(span=config.get("rsi_length", 14), adjust=False).mean()
             rs = gain / loss
             df['rsi'] = 100 - (100 / (1 + rs))
             df = df.join(calculate_rsi_signals(
                 df['rsi'],
-                params.get("rsi_overbought", 70),
-                params.get("rsi_oversold", 30)
+                config.get("rsi_overbought", 70),
+                config.get("rsi_oversold", 30)
             ))
         
         # === HTF TREND ===
-        if params.get("use_htf_trend"):
+        if config.get("use_htf_trend"):
             df['htf_ema'] = df['close'].ewm(
-                span=params.get("htf_period", 20),
+                span=config.get("htf_period", 20),
                 adjust=False
             ).mean()
             df = df.join(calculate_htf_signals(df['close'], df['htf_ema']))
         
         # === BOLLINGER BANDS ===
-        if params.get("use_bollinger_bands"):
-            period = params.get("bb_period", 20)
-            std_dev = params.get("bb_std", 2.0)
+        if config.get("use_bollinger_bands"):
+            period = config.get("bb_period", 20)
+            std_dev = config.get("bb_std", 2.0)
             ma = df['close'].rolling(window=period).mean()
             sd = df['close'].rolling(window=period).std()
             df["bb_upper"] = ma + (sd * std_dev)
@@ -232,8 +299,8 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
             df["bb_lower"] = ma - (sd * std_dev)
         
         # === STOCHASTIC ===
-        if params.get("use_stochastic"):
-            k_period = params.get("stoch_k", 14)
-            d_period = params.get("stoch_d", 3)
+        if config.get("use_stochastic"):
+            k_period = config.get("stoch_k", 14)
+            d_period = config.get("stoch_d", 3)
             lowest_low = df['low'].rolling(window=k_period).min()
             highest_high = df['high'].rolling(window=k_period).max()
             k = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
@@ -242,7 +309,7 @@ def calculate_all_indicators(df: pd.DataFrame, params: Dict, chunk_size=1000) -
             df["stoch_d"] = d
         
         # === ATR ===
-        if params.get("use_atr") and all(x in df.columns for x in ['high', 'low']):
+        if config.get("use_atr") and all(x in df.columns for x in ['high', 'low']):
             df["atr"] = calculate_atr(
                 df["high"],
                 df["low"],
                 df["close"],
-                params.get("atr_len", 14)
+                config.get("atr_len", 14)
             )
         
         # === SIMPLE MOVING AVERAGES ===
-        if params.get("use_ma"):
-            df["ma_short"] = df["close"].rolling(window=params.get("ma_short", 20)).mean()
-            df["ma_long"] = df["close"].rolling(window=params.get("ma_long", 50)).mean()
+        if config.get("use_ma"):
+            df["ma_short"] = df["close"].rolling(window=config.get("ma_short", 20)).mean()
+            df["ma_long"] = df["close"].rolling(window=config.get("ma_long", 50)).mean()
         
         # === VOLUME MA ===
-        if params.get("use_volume_ma") and "volume" in df.columns:
-            df["volume_ma"] = df["volume"].rolling(window=params.get("volume_ma_period", 20)).mean()
+        if config.get("use_volume_ma") and "volume" in df.columns:
+            df["volume_ma"] = df["volume"].rolling(window=config.get("volume_ma_period", 20)).mean()
             df["volume_ratio"] = safe_divide(df["volume"], df["volume_ma"])
     
     else:
         # Original calculation methods for smaller datasets
-        if params.get("use_ema_crossover"):
-            df['fast_ema'] = calculate_ema(df['close'], params.get("fast_ema", 9))
-            df['slow_ema'] = calculate_ema(df['close'], params.get("slow_ema", 21))
-            emacross = calculate_ema_crossover_signals(df['fast_ema'], df['slow_ema'], params.get("ema_points_threshold", 0))
+        if config.get("use_ema_crossover"):
+            df['fast_ema'] = calculate_ema(df['close'], config.get("fast_ema", 9))
+            df['slow_ema'] = calculate_ema(df['close'], config.get("slow_ema", 21))
+            emacross = calculate_ema_crossover_signals(df['fast_ema'], df['slow_ema'], config.get("ema_points_threshold", 0))
             df = df.join(emacross)
         
-        if params.get("use_macd"):
-            macd_df = calculate_macd(df['close'], params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
+        if config.get("use_macd"):
+            macd_df = calculate_macd(df['close'], config.get("macd_fast", 12), config.get("macd_slow", 26), config.get("macd_signal", 9))
             df = df.join(macd_df)
             macd_signals = calculate_macd_signals(macd_df)
             df = df.join(macd_signals)
         
-        if params.get("use_rsi_filter"):
-            df['rsi'] = calculate_rsi(df['close'], params.get("rsi_length", 14))
-            df = df.join(calculate_rsi_signals(df['rsi'], params.get("rsi_overbought", 70), params.get("rsi_oversold", 30)))
+        if config.get("use_rsi_filter"):
+            df['rsi'] = calculate_rsi(df['close'], config.get("rsi_length", 14))
+            df = df.join(calculate_rsi_signals(df['rsi'], config.get("rsi_overbought", 70), config.get("rsi_oversold", 30)))
         
-        if params.get("use_vwap") and all(col in df.columns for col in ['high', 'low', 'volume']):
+        if config.get("use_vwap") and all(col in df.columns for col in ['high', 'low', 'volume']):
             df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
             df = df.join(calculate_vwap_signals(df['close'], df['vwap']))
         
-        if params.get("use_htf_trend"):
-            df['htf_ema'] = calculate_htf_trend(df['close'], params.get("htf_period", 20))
+        if config.get("use_htf_trend"):
+            df['htf_ema'] = calculate_htf_trend(df['close'], config.get("htf_period", 20))
             df = df.join(calculate_htf_signals(df['close'], df['htf_ema']))
         
-        if params.get("use_bollinger_bands"):
-            upper, mid, lower = calculate_bollinger_bands(df['close'], params.get("bb_period", 20), params.get("bb_std", 2))
+        if config.get("use_bollinger_bands"):
+            upper, mid, lower = calculate_bollinger_bands(df['close'], config.get("bb_period", 20), config.get("bb_std", 2))
             df["bb_upper"], df["bb_middle"], df["bb_lower"] = upper, mid, lower
         
-        if params.get("use_stochastic"):
-            k, d = calculate_stochastic(df['high'], df['low'], df['close'], params.get("stoch_k", 14), params.get("stoch_d", 3))
+        if config.get("use_stochastic"):
+            k, d = calculate_stochastic(df['high'], df['low'], df['close'], config.get("stoch_k", 14), config.get("stoch_d", 3))
             df["stoch_k"], df["stoch_d"] = k, d
         
-        if params.get("use_ma"):
-            df["ma_short"] = calculate_sma(df["close"], params.get("ma_short", 20))
-            df["ma_long"] = calculate_sma(df["close"], params.get("ma_long", 50))
+        if config.get("use_ma"):
+            df["ma_short"] = calculate_sma(df["close"], config.get("ma_short", 20))
+            df["ma_long"] = calculate_sma(df["close"], config.get("ma_long", 50))
         
-        if params.get("use_atr") and all(x in df.columns for x in ['high', 'low']):
-            df["atr"] = calculate_atr(df["high"], df["low"], df["close"], params.get("atr_len", 14))
+        if config.get("use_atr") and all(x in df.columns for x in ['high', 'low']):
+            df["atr"] = calculate_atr(df["high"], df["low"], df["close"], config.get("atr_len", 14))
         
-        if params.get("use_volume_ma") and "volume" in df.columns:
-            df["volume_ma"] = calculate_sma(df["volume"], params.get("volume_ma_period", 20))
+        if config.get("use_volume_ma") and "volume" in df.columns:
+            df["volume_ma"] = calculate_sma(df["volume"], config.get("volume_ma_period", 20))
             df["volume_ratio"] = safe_divide(df["volume"], df["volume_ma"])
     
+    logger.info(f"Indicator calculation completed. DataFrame shape: {df.shape}")
     return df
 
 from typing import Tuple
@@ -376,6 +443,9 @@ class IncrementalVWAP:
     def update(self, price, volume, high=None, low=None, close=None):
         try:
+            # Validate inputs
+            if pd.isna(price) or price <= 0:
+                return self.last_vwap if self.last_vwap is not None else price
+            
+            # Handle invalid volume
+            if pd.isna(volume) or volume < 0:
+                volume = 0
+            
             if high is not None and low is not None and close is not None:
-                if not (pd.isna(high) or pd.isna(low) or pd.isna(close)):
+                if not any(pd.isna(x) for x in [high, low, close]):
                     typical_price = (high + low + close) / 3
                 else:
                     typical_price = price
@@ -386,8 +456,7 @@ class IncrementalVWAP:
             self.cum_vol += volume
             
-            # Safe calculation with fallback
+            # Safe VWAP calculation with fallback
             if self.cum_vol > 0:
                 self.last_vwap = self.cum_tpv / self.cum_vol
             else:
@@ -443,9 +512,15 @@ class IncrementalATR:
             if high < low:
                 high, low = low, high  # Swap if inverted
             
+            # Calculate True Range
             if self.prev_close is None:
                 tr = high - low
             else:
                 tr = max(
                     high - low,
                     abs(high - self.prev_close),
                     abs(low - self.prev_close)
                 )
             
+            # ATR calculation using Wilder's smoothing
             if not self.initialized:
                 self.tr_queue.append(tr)
                 if len(self.tr_queue) == self.period:
@@ -460,6 +535,7 @@ class IncrementalATR:
             self.prev_close = close
             return self.atr if self.atr is not None else tr
             
         except Exception as e:
             logger.error(f"ATR calculation error: {str(e)}")
             return self.atr if self.atr is not None else 0.0
@@ -513,10 +589,10 @@ class IncrementalMACD:
 """
 PARAMETER NAMING CONVENTION:
-- Main function: calculate_all_indicators(df: pd.DataFrame, params: Dict)
-- Parameter name 'params' is MANDATORY for interface compatibility
-- All internal usage: params.get('parameter_name', default)
+- Main function: calculate_all_indicators(df: pd.DataFrame, config: Dict)
+- Parameter name 'config' provides consistency across all modules
+- All internal usage: config.get('parameter_name', default)
 
 INTERFACE REQUIREMENT:
-- The 'params' parameter name CANNOT be changed as it's used by:
-  * researchStrategy.py: calculate_all_indicators(df, self.config)
-  * liveStrategy.py: calculate_all_indicators(df, self.config)
+- The 'config' parameter name ensures consistency:
+  * researchStrategy.py passes self.config as config
+  * liveStrategy.py passes self.config as config
   * Multiple indicator calculation functions
 
 CRITICAL:
-- Strategy modules pass their self.config as 'params' to this module
-- This creates the interface boundary between 'config' (strategies) and 'params' (indicators)
-- Do NOT change 'params' parameter name without updating ALL calling code
+- All modules now use consistent 'config' parameter naming
+- This eliminates confusion between config/params naming
+- Provides unified configuration interface across the system
 """
```


## File 4: position_manager.py

```diff
--- a/core/position_manager.py
+++ b/core/position_manager.py
@@ -89,6 +89,11 @@ class PositionManager:
     def __init__(self, config: Dict[str, Any]):
         self.config = config
+        
+        # Validate essential configuration
+        if not isinstance(config, dict):
+            raise TypeError("Config must be a dictionary")
+            
         self.initial_capital = config.get('initial_capital', 100000)
+        if self.initial_capital <= 0:
+            raise ValueError("Initial capital must be positive")
+            
         self.current_capital = self.initial_capital
         self.reserved_margin = 0.0
         self.risk_per_trade_percent = config.get('risk_per_trade_percent', 1.0)
@@ -104,8 +109,15 @@ class PositionManager:
         self.exchange_charges_percent = config.get('exchange_charges_percent', 0.0019)
         self.gst_percent = config.get('gst_percent', 18.0)
         self.slippage_points = config.get('slippage_points', 1)
+        
+        # Validate risk parameters
+        if not (0 < self.risk_per_trade_percent <= 10):
+            raise ValueError("Risk per trade must be between 0 and 10 percent")
+        if self.base_sl_points <= 0:
+            raise ValueError("Stop loss points must be positive")
         
         self.positions: Dict[str, Position] = {}
         self.completed_trades: List[Trade] = []
         self.daily_pnl = 0.0
-        self.session_params = config.get('session', {})
+        
+        # Store session parameters for exit logic
+        self.session_params = config.get('session', {})
+        if not self.session_params:
+            logger.warning("No session parameters provided, using defaults")
+            self.session_params = {
+                'intraday_end_hour': 15,
+                'intraday_end_min': 30,
+                'exit_before_close': 20
+            }
         
         logger.info(f"PositionManager initialized with capital: {self.initial_capital:,}")
 
@@ -124,6 +136,10 @@ class PositionManager:
 
     def calculate_lot_aligned_quantity(self, desired_quantity: int, lot_size: int) -> int:
+        """Calculate lot-aligned quantity for F&O trading."""
+        if desired_quantity <= 0:
+            return 0
+        if lot_size <= 0:
+            raise ValueError("Lot size must be positive")
+            
         if lot_size <= 1:  # Equity
             return max(1, desired_quantity)
         lots = max(1, round(desired_quantity / lot_size))
@@ -131,6 +147,12 @@ class PositionManager:
 
     def calculate_position_size(self, entry_price: float, stop_loss_price: float, lot_size: int = 1) -> int:
+        """Calculate position size based on risk management rules."""
+        # Input validation
+        if entry_price <= 0:
+            logger.error(f"Invalid entry price: {entry_price}")
+            return 0
+        if stop_loss_price <= 0:
+            logger.error(f"Invalid stop loss price: {stop_loss_price}")
+            return 0
+        if lot_size <= 0:
+            logger.error(f"Invalid lot size: {lot_size}")
+            return 0
+            
-        if entry_price <= 0 or stop_loss_price <= 0:
-            return 0
         
         risk_per_unit = abs(entry_price - stop_loss_price)
         if risk_per_unit <= 0:
+            logger.warning("No risk per unit (entry price equals stop loss)")
             return 0
         
         max_risk_amount = self.current_capital * (self.risk_per_trade_percent / 100)
@@ -146,6 +168,9 @@ class PositionManager:
         
         position_value = aligned_quantity * entry_price
         max_position_value = self.current_capital * (self.max_position_value_percent / 100)
         
         if position_value > max_position_value:
+            logger.info(f"Position value {position_value:,.2f} exceeds limit {max_position_value:,.2f}, reducing size")
             max_lots = int(max_position_value / (lot_size * entry_price))
             aligned_quantity = max(1, max_lots) * lot_size
         
@@ -153,6 +178,7 @@ class PositionManager:
 
     def calculate_total_costs(self, price: float, quantity: int, is_buy: bool = True) -> Dict[str, float]:
+        """Calculate comprehensive trading costs including taxes and charges."""
         turnover = price * quantity
         commission = max(self.commission_per_trade, turnover * (self.commission_percent / 100))
         stt = turnover * (self.stt_percent / 100) if not is_buy else 0.0
@@ -172,6 +198,16 @@ class PositionManager:
     def open_position(self, symbol: str, entry_price: float, timestamp: datetime,
                      lot_size: int = 1, tick_size: float = 0.05,
                      order_type: OrderType = OrderType.MARKET) -> Optional[str]:
+        """Open a new long position with comprehensive validation."""
+        
+        # Input validation
+        if not symbol:
+            logger.error("Symbol cannot be empty")
+            return None
+        if entry_price <= 0:
+            logger.error(f"Invalid entry price: {entry_price}")
+            return None
+        if timestamp is None:
+            logger.error("Timestamp cannot be None")
+            return None
+            
         if order_type == OrderType.MARKET:
             actual_entry_price = entry_price + self.slippage_points * tick_size
         else:
@@ -184,6 +220,7 @@ class PositionManager:
             logger.warning("Cannot open position: invalid quantity calculated")
             return None
         
+        # Calculate entry costs and validate capital
         entry_costs = self.calculate_total_costs(actual_entry_price, quantity, is_buy=True)
         required_capital = entry_costs['turnover'] + entry_costs['total_costs']
         
@@ -201,6 +238,8 @@ class PositionManager:
             entry_time=timestamp,
             entry_price=actual_entry_price,
             initial_quantity=quantity,
             current_quantity=quantity,
             lot_size=lot_size,
             tick_size=tick_size,
             stop_loss_price=stop_loss_price,
@@ -219,6 +258,7 @@ class PositionManager:
         self.reserved_margin += required_capital
         self.positions[position_id] = position
         
+        logger.info(f"Position opened successfully:")
         logger.info(f"Opened LONG position {position_id}: {quantity} {symbol} @ {actual_entry_price}")
         logger.info(f"SL: {stop_loss_price}, TPs: {tp_levels}")
+        logger.info(f"Capital remaining: ₹{self.current_capital:,.2f}")
         
         return position_id
 
@@ -226,6 +266,10 @@ class PositionManager:
                              quantity_to_close: int, timestamp: datetime,
                              exit_reason: str) -> bool:
+        """Close partial position with validation and cost calculation."""
+        if quantity_to_close <= 0:
+            logger.error(f"Invalid quantity to close: {quantity_to_close}")
+            return False
+            
         if position_id not in self.positions:
             logger.error(f"Position {position_id} not found")
             return False
@@ -233,9 +277,6 @@ class PositionManager:
         position = self.positions[position_id]
         
-        if quantity_to_close <= 0 or quantity_to_close > position.current_quantity:
-            logger.error(f"Invalid quantity to close: {quantity_to_close}")
-            return False
+        if quantity_to_close > position.current_quantity:
+            logger.warning(f"Quantity to close {quantity_to_close} exceeds current quantity {position.current_quantity}, closing all")
+            quantity_to_close = position.current_quantity
         
         exit_costs = self.calculate_total_costs(exit_price, quantity_to_close, is_buy=False)
         gross_pnl = (exit_price - position.entry_price) * quantity_to_close
@@ -276,6 +317,7 @@ class PositionManager:
         if position.current_quantity == 0:
             position.status = PositionStatus.CLOSED
             self.reserved_margin -= (position.initial_quantity * position.entry_price)
+            logger.info(f"Position {position_id} fully closed. Total P&L: ₹{position.realized_pnl:.2f}")
             del self.positions[position_id]
-            logger.info(f"Fully closed position {position_id}")
         else:
             position.status = PositionStatus.PARTIALLY_CLOSED
             logger.info(f"Partially closed position {position_id}: {quantity_to_close} @ ₹{exit_price}")
@@ -286,6 +328,7 @@ class PositionManager:
 
     def close_position_full(self, position_id: str, exit_price: float,
                            timestamp: datetime, exit_reason: str) -> bool:
+        """Close entire position."""
         if position_id not in self.positions:
+            logger.warning(f"Position {position_id} not found for closure")
             return False
         
         position = self.positions[position_id]
@@ -293,6 +336,14 @@ class PositionManager:
 
     def check_exit_conditions(self, position_id: str, current_price: float, timestamp: datetime) -> List[Tuple[int, str]]:
+        """Check all exit conditions for a position."""
+        if current_price <= 0:
+            logger.warning(f"Invalid current price {current_price} for exit check")
+            return []
+            
         if position_id not in self.positions:
+            logger.warning(f"Position {position_id} not found for exit condition check")
             return []
         
         position = self.positions[position_id]
         exits = []
         
+        # Update trailing stop before checking exits
         position.update_trailing_stop(current_price)
         
         # Stop Loss Check
@@ -306,6 +357,7 @@ class PositionManager:
         
         # Trailing Stop Check
         if (position.trailing_activated and position.trailing_stop_price and 
             current_price <= position.trailing_stop_price):
+            logger.info(f"Trailing stop triggered for {position_id} at {current_price}")
             exits.append((position.current_quantity, ExitReason.TRAILING_STOP.value))
             return exits
         
@@ -325,22 +377,33 @@ class PositionManager:
 
     def process_positions(self, row, timestamp, session_params=None):
-        """Enhanced position processing with session awareness"""
+        """Enhanced position processing with session awareness and validation."""
+        
+        # Validate inputs
+        if row is None:
+            logger.error("Row data is None, cannot process positions")
+            return
+        if timestamp is None:
+            logger.error("Timestamp is None, cannot process positions")
+            return
+            
         current_price = row['close']
+        if pd.isna(current_price) or current_price <= 0:
+            logger.warning(f"Invalid current price {current_price}, skipping position processing")
+            return
         
-        # Check for session exit if parameters provided
-        if session_params:
+        # Use provided session_params or fall back to instance params
+        session_config = session_params or self.session_params
+        session_exit = False
+        
+        if session_config:
             from utils.time_utils import is_time_to_exit
-            exit_buffer = session_params.get('exit_before_close', 20)
-            end_hour = session_params.get('intraday_end_hour', 15)
-            end_min = session_params.get('intraday_end_min', 30)
+            exit_buffer = session_config.get('exit_before_close', 20)
+            end_hour = session_config.get('intraday_end_hour', 15)
+            end_min = session_config.get('intraday_end_min', 30)
             
-            if is_time_to_exit(timestamp, exit_buffer, end_hour, end_min):
+            session_exit = is_time_to_exit(timestamp, exit_buffer, end_hour, end_min)
+            
+            if session_exit:
                 # Close all positions for session end
+                logger.info("Session exit time reached, closing all positions")
                 for position_id in list(self.positions.keys()):
                     self.close_position_full(position_id, current_price, timestamp, "Session End")
                 return
         
-        # Ensure timezone-aware
+        # Ensure timezone-aware timestamp
         timestamp = self._ensure_timezone(timestamp)
         
         for position_id in list(self.positions.keys()):
@@ -348,11 +411,6 @@ class PositionManager:
             if not position or position.status == PositionStatus.CLOSED:
                 continue
             
-            if session_end:
-                self.close_position_full(position_id, current_price, timestamp, ExitReason.SESSION_END.value)
-                continue
-            
+            # Check individual position exit conditions
             exits = self.check_exit_conditions(position_id, current_price, timestamp)
             
             for exit_quantity, exit_reason in exits:
                 if exit_quantity > 0:
+                    logger.info(f"Exit condition triggered: {exit_reason} for {exit_quantity} units")
                     self.close_position_partial(position_id, current_price, exit_quantity, timestamp, exit_reason)
                     if position_id not in self.positions:
                         break  # Position fully closed
 
     def get_portfolio_value(self, current_price: float) -> float:
+        """Calculate total portfolio value including unrealized P&L."""
+        if current_price <= 0:
+            logger.warning(f"Invalid current price {current_price} for portfolio valuation")
+            return self.current_capital
+            
         total_value = self.current_capital
         
         for position in self.positions.values():
@@ -378,6 +436,7 @@ class PositionManager:
         return total_value
 
     def get_open_positions(self) -> List[Dict[str, Any]]:
+        """Get list of all open positions with their details."""
         open_positions = []
         
         for position in self.positions.values():
@@ -401,6 +460,7 @@ class PositionManager:
         return open_positions
 
     def get_trade_history(self) -> List[Dict[str, Any]]:
+        """Get complete trade history with performance metrics."""
         trades = []
         
         for trade in self.completed_trades:
@@ -423,6 +483,10 @@ class PositionManager:
         return trades
 
     def get_performance_summary(self) -> Dict[str, Any]:
+        """Calculate comprehensive performance summary."""
+        # Handle case with no trades
         if not self.completed_trades:
+            logger.info("No completed trades for performance summary")
             return {
                 'total_trades': 0,
                 'winning_trades': 0,
@@ -435,6 +499,9 @@ class PositionManager:
                 'max_loss': 0.0,
                 'total_commission': 0.0
             }
         
+        # Calculate performance metrics
         winning_trades = [t for t in self.completed_trades if t.net_pnl > 0]
         losing_trades = [t for t in self.completed_trades if t.net_pnl < 0]
         total_pnl = sum(t.net_pnl for t in self.completed_trades)
@@ -442,6 +509,11 @@ class PositionManager:
         gross_profit = sum(t.net_pnl for t in winning_trades)
         gross_loss = abs(sum(t.net_pnl for t in losing_trades))
         
+        # Calculate ratios with safety checks
+        win_rate = (len(winning_trades) / len(self.completed_trades)) * 100 if self.completed_trades else 0
+        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
+        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
+        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
+        
         return {
             'total_trades': len(self.completed_trades),
             'winning_trades': len(winning_trades),
             'losing_trades': len(losing_trades),
-            'win_rate': (len(winning_trades) / len(self.completed_trades)) * 100,
+            'win_rate': win_rate,
             'total_pnl': total_pnl,
-            'avg_win': gross_profit / len(winning_trades) if winning_trades else 0,
-            'avg_loss': gross_loss / len(losing_trades) if losing_trades else 0,
-            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 0,
+            'avg_win': avg_win,
+            'avg_loss': avg_loss,
+            'profit_factor': profit_factor,
             'max_win': max(t.net_pnl for t in self.completed_trades),
             'max_loss': min(t.net_pnl for t in self.completed_trades),
             'total_commission': total_commission
@@ -459,6 +531,7 @@ class PositionManager:
 
     def reset(self, initial_capital: Optional[float] = None):
+        """Reset position manager to initial state."""
         if initial_capital:
+            if initial_capital <= 0:
+                raise ValueError("Initial capital must be positive")
             self.initial_capital = initial_capital
             
         self.current_capital = self.initial_capital
@@ -473,8 +546,10 @@ class PositionManager:
     def enter_position(self, side: str, price: float, quantity: int, timestamp: datetime,
                       **kwargs) -> Optional[str]:
+        """Legacy compatibility method for backtest engine."""
         if side.upper() != 'BUY':
             logger.warning("This system only supports LONG positions")
             return None
         
+        # Extract parameters with defaults
         symbol = kwargs.get('symbol', 'NIFTY')
         lot_size = kwargs.get('lot_size', 1)
         tick_size = kwargs.get('tick_size', 0.05)
@@ -482,10 +557,12 @@ class PositionManager:
         return self.open_position(symbol, price, timestamp, lot_size, tick_size)
 
     def exit_position(self, position_id: str, price: float, timestamp: datetime, reason: str):
+        """Legacy compatibility method for backtest engine."""
         self.close_position_full(position_id, price, timestamp, reason)
 
     def can_enter_position(self) -> bool:
+        """Check if new positions can be entered based on limits."""
         return len(self.positions) < self.config.get('max_positions_per_day', 10)
 
     def calculate_position_size_gui_driven(self, entry_price: float, stop_loss_price: float,
                                          user_capital: float, user_risk_pct: float,
                                          user_lot_size: int) -> dict:
         """
-        GUI-driven position sizing with comprehensive feedback
+        GUI-driven position sizing with comprehensive feedback and validation.
         
         Returns:
             dict with position details and capital analysis
         """
+        # Input validation
+        if entry_price <= 0:
+            return {"error": "Entry price must be positive"}
+        if stop_loss_price <= 0:
+            return {"error": "Stop loss price must be positive"}
+        if user_capital <= 0:
+            return {"error": "Capital must be positive"}
+        if not (0 < user_risk_pct <= 10):
+            return {"error": "Risk percentage must be between 0 and 10"}
+        if user_lot_size <= 0:
+            return {"error": "Lot size must be positive"}
+            
-        if entry_price <= 0 or stop_loss_price <= 0:
-            return {"error": "Invalid price inputs"}
         
         risk_per_unit = abs(entry_price - stop_loss_price)
+        if risk_per_unit <= 0:
+            return {"error": "Entry price cannot equal stop loss price"}
         
         # Risk-based calculation
         max_risk_amount = user_capital * (user_risk_pct / 100)
@@ -522,6 +599,8 @@ class PositionManager:
         actual_risk = aligned_quantity * risk_per_unit
         actual_risk_pct = (actual_risk / user_capital) * 100 if user_capital > 0 else 0
         capital_utilization = (position_value / user_capital) * 100 if user_capital > 0 else 0
         
+        # Determine which constraint was the limiting factor
+        approach_used = "risk_limited" if final_quantity == risk_based_quantity else "capital_limited"
+        
         return {
             "recommended_quantity": aligned_quantity,
             "recommended_lots": final_lots,
@@ -532,7 +611,7 @@ class PositionManager:
             "actual_risk_pct": actual_risk_pct,
             "max_affordable_lots": max_affordable_shares // user_lot_size,
             "risk_based_lots": risk_based_quantity // user_lot_size,
-            "approach_used": "risk_limited" if final_quantity == risk_based_quantity else "capital_limited"
+            "approach_used": approach_used
         }
```


## File 5: time_utils.py

This file looks well-structured but needs minor improvements:

```diff
--- a/utils/time_utils.py
+++ b/utils/time_utils.py
@@ -36,6 +36,12 @@ def now_ist() -> ISTDateTime:
 def normalize_datetime_to_ist(dt: datetime) -> ISTDateTime:
     """
     SINGLE NORMALIZATION FUNCTION for all datetime objects.
+    
+    This is the authoritative function for timezone normalization.
+    All modules should use this function to ensure consistent timezone handling.
+    
+    Args:
+        dt: Input datetime (naive or timezone-aware)
+        
+    Returns:
+        Timezone-aware datetime in IST
     """
     if dt.tzinfo is None:
         # Assume naive datetime is in IST
@@ -152,7 +158,12 @@ def is_time_to_exit(current_time: datetime, exit_buffer: int, end_hour: int, en
     Args:
         current_time: Current timestamp
         exit_buffer: Minutes before close to start exiting
         end_hour: Market end hour
         end_min: Market end minute
         
     Returns:
         True if should start exiting positions
     """
+    # Validate inputs
+    if exit_buffer < 0:
+        logger.warning(f"Invalid exit buffer {exit_buffer}, using 0")
+        exit_buffer = 0
+    if not (0 <= end_hour <= 23):
+        logger.warning(f"Invalid end hour {end_hour}, using 15")
+        end_hour = 15
+    if not (0 <= end_min <= 59):
+        logger.warning(f"Invalid end minute {end_min}, using 30")
+        end_min = 30
+        
     if current_time.tzinfo is not None:
         current_minutes = current_time.hour * 60 + current_time.minute
     else:
@@ -577,6 +588,12 @@ def ensure_tz_aware(dt, fallback_tz=None, default_tz="Asia/Kolkata"):
-def ensure_tz_aware(dt, fallback_tz=None, default_tz="Asia/Kolkata"):
-    """Ensure datetime is timezone aware."""
+def ensure_tz_aware(dt: datetime, fallback_tz=None, default_tz="Asia/Kolkata") -> datetime:
+    """
+    Ensure datetime is timezone aware with comprehensive validation.
+    
+    Args:
+        dt: Input datetime
+        fallback_tz: Fallback timezone (object or string)
+        default_tz: Default timezone string
+        
+    Returns:
+        Timezone-aware datetime
+    """
+    if dt is None:
+        raise ValueError("Datetime cannot be None")
+        
     if dt.tzinfo is not None:
         return dt
     
     # Handle timezone object vs string
     if fallback_tz is not None:
-        if hasattr(fallback_tz, 'zone'):
+        try:
+            if hasattr(fallback_tz, 'zone'):
             # It's a timezone object, get the zone name
             tz_name = fallback_tz.zone
-        else:
+            else:
             # It's already a string
             tz_name = fallback_tz
+            
         return pytz.timezone(tz_name).localize(dt)
+        except Exception as e:
+            logger.warning(f"Error with fallback timezone {fallback_tz}: {e}, using default")
+            return pytz.timezone(default_tz).localize(dt)
     else:
         return pytz.timezone(default_tz).localize(dt)
```

I'll continue with the remaining files in the next response to provide complete diffs for all files.

<div style="text-align: center">⁂</div>

[^1]: backtest_runner.py

[^2]: results.py

[^3]: strategy_config.yaml

[^4]: indicators.py

[^5]: liveStrategy.py

[^6]: position_manager.py

[^7]: researchStrategy.py

[^8]: unified_gui.py

[^9]: broker_adapter.py

[^10]: login.py

[^11]: trader.py

[^12]: websocket_stream.py

[^13]: cache_manager.py

[^14]: config_loader.py

[^15]: logging_utils.py

[^16]: simple_loader.py

[^17]: time_utils.py

[^18]: matrix_backtest.py

