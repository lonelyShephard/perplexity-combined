# backtest_runner_refactored.py
"""
Refactored backtest runner implementing "normalize early, standardize everywhere" principle.

Key Changes:
- All data normalization moved to single entry point
- Uses centralized DataNormalizer for consistent data quality
- Downstream components assume data is pre-validated
- Improved error handling and logging
- Better separation of concerns

Data Flow:
Raw Data → DataNormalizer → Indicators → Strategy → Position Manager
          ↑
    SINGLE SOURCE OF TRUTH
    (All normalization happens here)

    """

import yaml
import importlib
import logging
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, Tuple
import inspect

from core.position_manager import PositionManager
from utils.simple_loader import load_data_simple
from utils.time_utils import normalize_datetime_to_ist, now_ist, ensure_tz_aware, is_time_to_exit, is_trading_session
from utils.config_helper import ConfigAccessor
import logging
from core.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

try:
    # Verify the function exists and comes from time_utils
    assert callable(ensure_tz_aware)
    assert 'time_utils' in inspect.getmodule(ensure_tz_aware).__name__
    logger.info(f"✅ ensure_tz_aware verified from {inspect.getmodule(ensure_tz_aware).__name__}")
except (AssertionError, AttributeError, ImportError) as e:
    logger.error(f"❌ ensure_tz_aware verification failed: {e}")
    # Raise immediately to prevent hard-to-diagnose errors later
    raise ImportError("Critical timezone function not properly available")

def load_config(config_path: str) -> dict:
    """Load YAML strategy and risk config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_strategy(config: dict):
    """
    Load strategy module with full configuration.

    FIXED: Properly handles GUI's nested config structure
    """
    version = config.get("strategy", {}).get("strategy_version", "live").lower()

    if version == "research":
        strat_mod = importlib.import_module("core.researchStrategy")
    else:
        strat_mod = importlib.import_module("core.liveStrategy")
     
    ind_mod = importlib.import_module("core.indicators")
    
    # FIXED: Keep nested structure, no more flattening
    logger.info("NESTED CONFIG: Using consistent nested configuration structure")
    logger.info(f"Strategy parameters found: {list(config.get('strategy', {}).keys())}")
     
    # Pass nested config directly to strategy
    return strat_mod.ModularIntradayStrategy(config, ind_mod)

def run_backtest(config: Dict[str, Any], data_file: str,
                 df_normalized=None, skip_indicator_calculation=False):
    """Run a backtest with the given configuration"""
    logger.info("=" * 60)
    logger.info("STARTING BACKTEST WITH NORMALIZED DATA PIPELINE")
    logger.info("=" * 60)
    
    # Load configuration
    if isinstance(config, str):
        config = load_config(config)
        
    # Extract parameters
    strategy_params = config.get('strategy', {})
    session_params = config.get('session', {})
    risk_params = config.get('risk', {})
    instrument_params = config.get('instrument', {})
    capital = config.get('capital', {}).get('initial_capital', 100000)
    
    # Add a default for the symbol if it's missing
    if 'symbol' not in instrument_params:
        instrument_params['symbol'] = 'DEFAULT_SYMBOL'
        logger.warning(f"Instrument symbol not found in config. Using default: '{instrument_params['symbol']}'")
     
    # FIXED: Maintain consistent nested structure throughout
    logger.info("=== NESTED CONFIG STRUCTURE MAINTAINED ===")
    for section, params in config.items():
        if isinstance(params, dict):
            logger.info(f"Section '{section}': {len(params)} parameters")
     
    # Ensure session parameters are consistent
    session_params = config.get("session", {})
    if "intraday_end_min" not in session_params:
        session_params["intraday_end_min"] = 30  # Consistent with NSE close time
    if "exit_before_close" not in session_params:
        session_params["exit_before_close"] = 20  # Default value
    if "timezone" not in session_params:
        session_params["timezone"] = "Asia/Kolkata"

    # Initialize components with nested config
    strategy = get_strategy(config)
    
    # FIXED: Pass nested config directly to PositionManager
    logger.info("=== NESTED CONFIG PASSED TO POSITION MANAGER ===")
    logger.info(f"Config sections: {list(config.keys())}")
    
    # Validate critical sections exist
    required_sections = ['strategy', 'risk', 'capital', 'instrument', 'session']
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        logger.warning(f"❌ MISSING config sections: {missing_sections}")
     
    # Initialize PositionManager with nested config
    position_manager = PositionManager(config)
    
    # Skip data loading if df_normalized is provided
    if df_normalized is None:
        logger.info("Loading data with centralized loader...")
        df_normalized, quality_report = load_and_normalize_data(data_file, process_as_ticks=True)
        logger.info(f"Loaded and normalized data. Shape: {df_normalized.shape}. Time range: {df_normalized.index.min()} to {df_normalized.index.max()}")
        if df_normalized.empty:
            logger.error("CRITICAL: DataFrame is empty after normalization. Cannot proceed.")
            return pd.DataFrame(), position_manager.get_performance_summary()
    else:
        # If df_normalized is provided, create a simple quality report
        # Calculate sample indices even for pre-loaded dataframes
        sample_indices = []
        total_rows = len(df_normalized)
        for chunk_start in range(0, total_rows, 1000):
            chunk_end = min(chunk_start + 1000, total_rows)
            if chunk_end - chunk_start >= 5:
                step = (chunk_end - chunk_start) // 5
                chunk_sample = [chunk_start + i * step for i in range(5)]
            else:
                chunk_sample = list(range(chunk_start, chunk_end))
            sample_indices.extend(chunk_sample)
        
        # Remove duplicates and ensure indices are within bounds
        sample_indices = sorted(list(set(idx for idx in sample_indices if idx < total_rows)))
        
        # Create quality report WITH sample_indices
        quality_report = type('SimpleQualityReport', (), {
            'total_rows': len(df_normalized),
            'rows_processed': len(df_normalized),
            'rows_dropped': 0,
            'issues_found': {},
            'sample_indices': sample_indices  # Add this critical field
        })
    
    # Optimize memory usage for large tick dataset
    if len(df_normalized) > 5000:
        logger.info(f"Optimizing memory usage for large tick dataset ({len(df_normalized)} ticks)")
        # Chunk processing: Process full dataset in memory-efficient chunks
        chunk_size = min(2000, max(1000, len(df_normalized) // 10))  # Adaptive chunk size
        logger.info(f"Processing {len(df_normalized)} rows in chunks of {chunk_size}")
        
        # Process indicators in chunks and combine results
        df_with_indicators = process_indicators_sequential(df_normalized, strategy, chunk_size)
        logger.info(f"Chunk processing completed. Full dataset with indicators: {len(df_with_indicators)} rows")
    else:
        # Normal processing for smaller datasets
        df_with_indicators = strategy.calculate_indicators(df_normalized)
    
    # === STAGE 3: AFTER INDICATOR CALCULATION ===
    if hasattr(quality_report, 'sample_indices'):
        logger.info("=" * 80)
        logger.info("STAGE 3: AFTER INDICATOR CALCULATION (Same Rows)")
        logger.info("=" * 80)
        
        for i, idx in enumerate(quality_report.sample_indices[:25]):
            if idx < len(df_with_indicators):
                row_data = df_with_indicators.iloc[idx]
                
                # Build indicator summary string
                indicators = []
                if 'fast_ema' in row_data and not pd.isna(row_data['fast_ema']):
                    indicators.append(f"FastEMA={row_data['fast_ema']:.3f}")
                if 'slow_ema' in row_data and not pd.isna(row_data['slow_ema']):
                    indicators.append(f"SlowEMA={row_data['slow_ema']:.3f}")
                if 'vwap' in row_data and not pd.isna(row_data['vwap']):
                    indicators.append(f"VWAP={row_data['vwap']:.3f}")
                if 'macd' in row_data and not pd.isna(row_data['macd']):
                    indicators.append(f"MACD={row_data['macd']:.4f}")
                if 'rsi' in row_data and not pd.isna(row_data['rsi']):
                    indicators.append(f"RSI={row_data['rsi']:.1f}")
                
                indicator_str = ", ".join(indicators[:4]) if indicators else "No indicators"
                
                # Signal status
                signals = []
                if 'ema_bullish' in row_data:
                    signals.append(f"EMA_Bull={row_data['ema_bullish']}")
                if 'vwap_bullish' in row_data:
                    signals.append(f"VWAP_Bull={row_data['vwap_bullish']}")
                
                signal_str = ", ".join(signals) if signals else "No signals"
                
                logger.info(f"Ind  Row {idx:6d} (Sample {i+1:2d}): "
                           f"Time={row_data.name}, "
                           f"Close={row_data.get('close', 'N/A'):8.2f}, "
                           f"[{indicator_str}], Signals=[{signal_str}]")
    
    # Log final indicator status for verification
    logger.info("Indicators calculated. Final 5 rows:")
    if 'fast_ema' in df_with_indicators.columns and 'slow_ema' in df_with_indicators.columns:
        logger.info(f"\n{df_with_indicators[['close', 'fast_ema', 'slow_ema', 'vwap']].tail(5).to_string()}")
    else:
        logger.info(f"\n{df_with_indicators[['close', 'volume']].tail(5).to_string()}")

    # Quick EMA diagnostic
    if 'fast_ema' in df_with_indicators.columns:
        fast_above_slow = (df_with_indicators['fast_ema'] > df_with_indicators['slow_ema']).sum()
        total_rows = len(df_with_indicators)
        logger.info(f"EMA DIAGNOSTIC: {fast_above_slow} out of {total_rows} rows have fast > slow ({fast_above_slow/total_rows*100:.1f}%)")
        # Show a sample of EMA values
        sample = df_with_indicators[['fast_ema', 'slow_ema']].dropna().head(10)
        logger.info(f"Sample EMA values:\n{sample.to_string()}")

    # Backtest execution loop
    logger.info("Starting backtest execution...")
    position_id = None
    in_position = False
    
    # Extract session parameters for exit logic (CORRECTED)
    session_params = config.get('session', {})  # ✅ Fixed: consistent config naming
    close_hour = session_params.get("intraday_end_hour", 15)
    close_min = session_params.get("intraday_end_min", 30)  # ✅
    exit_buffer = session_params.get("exit_before_close", 20)

    processed_bars = 0  # Add this counter
    
    for timestamp, row in df_with_indicators.iterrows():  # ✅ Fixed: df_with_indicators not df_ind
        processed_bars += 1
        
        # ENSURE timezone awareness for timestamp
        now = ensure_tz_aware(timestamp)

        # Add this after normalizing the timestamp
        row['session_exit'] = is_time_to_exit(now, exit_buffer, close_hour, close_min)

        # Check if in exit buffer period - CORE LOGIC (move this up)
        if is_time_to_exit(now, exit_buffer, close_hour, close_min):
            # Close all positions and terminate
            for pos_id in list(position_manager.positions.keys()):
                position_manager.close_position_full(pos_id, row['close'], now, "Exit Buffer")
            break  # Stop processing completely

        # For debugging the first few iterations
        if processed_bars <= 1:
            logger.info(f"Processing timestamp: {now} (tzinfo: {now.tzinfo})")
        
        # Process positions with timezone-aware timestamp
        position_manager.process_positions(row, now)
        
        # Entry Logic: only if not already in position and conditions meet
        if not in_position and strategy.can_open_long(row, now):
            position_id = strategy.open_long(row, now, position_manager)
            in_position = position_id is not None
            
            if in_position:
                logger.debug(f"Opened position {position_id} at {now} @ {row['close']:.2f}")
        
        # Exit Logic: PositionManager handles trailing stops, TPs, SLs and session-end exits
        if in_position:
            position_manager.process_positions(row, now)
            
            # Check for strategy-level exit conditions
            if strategy.should_close(row, now, position_manager):
                last_price = row['close']
                strategy.handle_exit(position_id, last_price, now, position_manager, reason="Strategy Exit")
                in_position = False
                position_id = None
                logger.debug(f"Strategy exit at {now} @ {last_price:.2f}")
        else:
            # Still allow PositionManager to process positions in edge cases
            position_manager.process_positions(row, now)
        
        # Reset position state if position closed by PositionManager
        if position_id and position_id not in position_manager.positions:
            in_position = False
            position_id = None
        
        # Log first timestamp info
        if processed_bars == 1:
            # Log details of the first timestamp to verify timezone awareness
            logger.info(f"First timestamp processing details:")
            logger.info(f"  - Original timestamp: {timestamp} (tzinfo: {timestamp.tzinfo})")
            logger.info(f"  - Normalized 'now': {now} (tzinfo: {now.tzinfo})")
            logger.info(f"  - Session exit check: {row['session_exit']}")
            # Check if strategy methods handle the timestamp properly
            logger.info(f"  - Session live check: {strategy.is_session_live(now)}")
    
    # Defensive: flatten any still-open positions at backtest end
    if position_id and position_id in position_manager.positions:
        last_price = df_with_indicators.iloc[-1]['close']
        now = df_with_indicators.index[-1]
        strategy.handle_exit(position_id, last_price, now, position_manager, reason="End of Backtest")
        logger.info(f"Closed final position at backtest end @ {last_price:.2f}")
    
    # Gather and print summary
    trades = position_manager.get_trade_history()
    performance = position_manager.get_performance_summary()
    
    logger.info("=" * 60)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data Quality:")
    logger.info(f"  Total input rows: {quality_report.total_rows}")
    logger.info(f"  Processed rows: {quality_report.rows_processed}")
    logger.info(f"  Data quality: {quality_report.rows_processed/quality_report.total_rows*100:.1f}%")
    logger.info("")
    logger.info(f"Trading Performance:")
    logger.info(f"  Total Trades: {performance['total_trades']}")
    logger.info(f"  Win Rate: {performance['win_rate']:.2f}%")
    logger.info(f"  Total P&L: {performance['total_pnl']:.2f}")
    logger.info(f"  Avg Win: {performance['avg_win']:.2f}")
    logger.info(f"  Avg Loss: {performance['avg_loss']:.2f}")
    logger.info(f"  Profit Factor: {performance['profit_factor']:.2f}")
    logger.info(f"  Max Win: {performance['max_win']:.2f}")
    logger.info(f"  Max Loss: {performance['max_loss']:.2f}")
    logger.info(f"  Total Commission: {performance['total_commission']:.2f}")
    logger.info("=" * 60)
    
    # Save trade log CSV file
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv("backtest_trades.csv", index=False)
        logger.info("Trade log written to backtest_trades.csv")
    else:
        logger.warning("No trades executed during backtest")
        trades_df = pd.DataFrame()
    
    return trades_df, performance

def run_backtest_debug(strategy, data, position_manager, risk_manager, start_date, end_date):
    """Enhanced backtest with production-level debugging."""
    
    # Add at the beginning of the method
    logger.info(f"Starting backtest with {len(data)} rows")
    logger.info(f"Strategy type: {type(strategy)}")
    
    # Verify strategy has required methods
    required_methods = ['can_open_long', 'open_long', 'calculate_indicators']
    missing_methods = [m for m in required_methods if not hasattr(strategy, m)]
    if missing_methods:
        logger.error(f"CRITICAL: Strategy missing methods: {missing_methods}")
        return {}
    
    signals_checked = 0
    entries_attempted = 0
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        try:
            # Debug every 100 rows
            if i % 100 == 0:
                logger.info(f"Processing row {i}/{len(data)}: {timestamp}")
            
            # Check if strategy can open long
            can_open = strategy.can_open_long(row, timestamp)
            signals_checked += 1
            
            if can_open:
                logger.info(f"SIGNAL DETECTED at {timestamp}: Price={row['close']}")
                entries_attempted += 1
                
                # Attempt to open position
                position_id = strategy.open_long(row, timestamp, position_manager)
                if position_id:
                    logger.info(f"TRADE EXECUTED: Position {position_id} opened")
                else:
                    logger.warning(f"TRADE FAILED: Could not open position")
            
            # Add this debug every 100 rows
            if i % 100 == 0:
                logger.info(f"Processed {i} rows, Signals checked: {signals_checked}, Entries attempted: {entries_attempted}")
        
        except Exception as e:
            logger.error(f"Error processing row {i}: {e}")
            continue
    
    logger.info("Backtest debug completed")
    return {}

def load_and_normalize_data(data_path: str, process_as_ticks: bool = False) -> Tuple[pd.DataFrame, Any]:
    """
    Centralized data loading function with comprehensive row tracking.
    """
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # === STAGE 1: RAW DATA LOADING ===
    df_raw = load_data_simple(data_path, process_as_ticks)
    
    # Calculate sample rows (5 rows every 1000 rows)
    total_rows = len(df_raw)
    sample_indices = []
    
    for chunk_start in range(0, total_rows, 1000):
        chunk_end = min(chunk_start + 1000, total_rows)
        # Select 5 evenly distributed rows within each 1000-row chunk
        if chunk_end - chunk_start >= 5:
            step = (chunk_end - chunk_start) // 5
            chunk_sample = [chunk_start + i * step for i in range(5)]
        else:
            # If chunk has less than 5 rows, take all
            chunk_sample = list(range(chunk_start, chunk_end))
        
        sample_indices.extend(chunk_sample)
    
    # Remove duplicates and ensure indices are within bounds
    sample_indices = sorted(list(set(idx for idx in sample_indices if idx < total_rows)))
    
    logger.info("=" * 80)
    logger.info("STAGE 1: RAW DATA SAMPLE (5 rows per 1000)")
    logger.info("=" * 80)
    logger.info(f"Sampling {len(sample_indices)} rows from {total_rows} total rows")
    
    for i, idx in enumerate(sample_indices[:25]):  # Limit output for readability
        row_data = df_raw.iloc[idx]
        logger.info(f"Raw Row {idx:6d} (Sample {i+1:2d}): "
                   f"Time={row_data.name}, "
                   f"Close={row_data.get('close', 'N/A'):8.2f}, "
                   f"Volume={row_data.get('volume', 'N/A'):6.0f}")
    
    # === STAGE 2: AFTER NORMALIZATION ===
    df_normalized = df_raw  # Your normalization happens in simple_loader
    
    logger.info("=" * 80)
    logger.info("STAGE 2: AFTER NORMALIZATION (Same Rows)")
    logger.info("=" * 80)
    
    for i, idx in enumerate(sample_indices[:25]):
        if idx < len(df_normalized):
            row_data = df_normalized.iloc[idx]
            logger.info(f"Norm Row {idx:6d} (Sample {i+1:2d}): "
                       f"Time={row_data.name}, "
                       f"Close={row_data.get('close', 'N/A'):8.2f}, "
                       f"Volume={row_data.get('volume', 'N/A'):6.0f}")
    
    # Store sample indices for later use in indicator stage
    quality_report = type('DetailedQualityReport', (), {
        'total_rows': len(df_normalized),
        'rows_processed': len(df_normalized),
        'rows_dropped': 0,
        'issues_found': {},
        'sample_indices': sample_indices
    })
    
    # Add basic validation (previously handled by DataNormalizer)
    if df_normalized.isnull().any().any():
        logger.warning(f"Dataset contains {df_normalized.isnull().sum().sum()} missing values")
    
    # Check for negative prices (previously handled by DataNormalizer)
    neg_prices = (df_normalized['close'] <= 0).sum() if 'close' in df_normalized.columns else 0
    if neg_prices > 0:
        logger.warning(f"Dataset contains {neg_prices} negative or zero prices")
    
    logger.info("=== COMPLETE DATASET ANALYSIS ===")
    logger.info(f"Dataset shape: {df_normalized.shape}")
    logger.info(f"Time range: {df_normalized.index.min()} to {df_normalized.index.max()}")
    logger.info(f"Total duration: {df_normalized.index.max() - df_normalized.index.min()}")

    # Show time distribution
    time_groups = df_normalized.groupby(df_normalized.index.hour).size()
    logger.info("Hourly tick distribution:")
    for hour, count in time_groups.items():
        logger.info(f"  Hour {hour:02d}: {count:,} ticks")

    # Show first and last 10 rows with timestamps
    logger.info("First 10 rows:")
    logger.info(f"\n{df_normalized.head(10)[['close', 'volume']].to_string()}")
    logger.info("Last 10 rows:")
    logger.info(f"\n{df_normalized.tail(10)[['close', 'volume']].to_string()}")
    
    return df_normalized, quality_report

def add_indicator_signals_to_chunk(chunk_df: pd.DataFrame, config: Dict[str, Any]):
    """
    Add indicator signals to a processed chunk.
    
    Args:
        chunk_df: DataFrame chunk with computed indicators
        config: Strategy configuration
    """
    from core.indicators import (
        calculate_ema_crossover_signals, calculate_macd_signals, 
        calculate_vwap_signals, calculate_htf_signals, calculate_rsi_signals
    )
    
    # EMA Crossover Signals
    if config.get('use_ema_crossover', False) and 'fast_ema' in chunk_df.columns:
        ema_signals = calculate_ema_crossover_signals(
            chunk_df['fast_ema'],
            chunk_df['slow_ema'],
        )
        chunk_df = chunk_df.join(ema_signals)
    
    # MACD Signals
    if config.get('use_macd', False) and 'macd' in chunk_df.columns:
        macd_df = pd.DataFrame({
            'macd': chunk_df['macd'],
            'signal': chunk_df['macd_signal'],
            'histogram': chunk_df['histogram']
        })
        macd_signals = calculate_macd_signals(macd_df)
        chunk_df = chunk_df.join(macd_signals)
    
    # VWAP Signals
    if config.get('use_vwap', False) and 'vwap' in chunk_df.columns:
        vwap_signals = calculate_vwap_signals(chunk_df['close'], chunk_df['vwap'])
        chunk_df = chunk_df.join(vwap_signals)
    
    # HTF Trend Signals
    if config.get('use_htf_trend', False) and 'htf_ema' in chunk_df.columns:
        htf_signals = calculate_htf_signals(chunk_df['close'], chunk_df['htf_ema'])
        chunk_df = chunk_df.join(htf_signals)
    
    # RSI Signals
    if config.get('use_rsi_filter', False) and 'rsi' in chunk_df.columns:
        rsi_signals = calculate_rsi_signals(
            chunk_df['rsi'],
            config.get('rsi_overbought', 70),
            config.get('rsi_oversold', 30)
        )
        chunk_df = chunk_df.join(rsi_signals)
    
    return chunk_df

def process_indicators_sequential(df_normalized: pd.DataFrame, strategy, chunk_size: int = 2000) -> pd.DataFrame:
    """
    Process indicators sequentially without overlapping chunks to eliminate data corruption.

    This approach:
    1. Processes data in non-overlapping sequential chunks
    2. Uses stateful indicators that maintain continuity across chunks  
    3. Combines results without complex index manipulation
    4. Eliminates the risk of data corruption from overlapping windows
    """
    logger.info("Starting sequential chunk-based indicator processing...")

    total_rows = len(df_normalized)

    # For small datasets, process normally without chunking
    if total_rows <= chunk_size:
        logger.info(f"Small dataset ({total_rows} rows), processing without chunking")
        return strategy.calculate_indicators(df_normalized)

    # Setup diagnostics
    logger.info("=== CHUNK PROCESSING DIAGNOSTICS ===")
    logger.info(f"Input dataset: {len(df_normalized)} rows")
    logger.info(f"Chunk size: {chunk_size}")
    logger.info(f"Number of chunks: {(len(df_normalized) + chunk_size - 1) // chunk_size}")
    
    processed_chunks = []
    chunk_summaries = []
    
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    for chunk_num in range(1, num_chunks + 1):
        start_idx = (chunk_num - 1) * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_df = df_normalized.iloc[start_idx:end_idx].copy()

        logger.info(f"Processing chunk {chunk_num}: rows {start_idx}-{end_idx}")

        try:
            # Pass nested config directly to calculate_all_indicators
            chunk_with_indicators = calculate_all_indicators(chunk_df, strategy.config)

            chunk_summary = {
                'chunk': chunk_num,
                'rows': len(chunk_with_indicators),
                'time_start': chunk_with_indicators.index[0],
                'time_end': chunk_with_indicators.index[-1],
                'ema_crossovers': 0,
                'vwap_bullish': 0
            }

            if 'fast_ema' in chunk_with_indicators.columns and 'slow_ema' in chunk_with_indicators.columns:
                ema_cross = (chunk_with_indicators['fast_ema'] > chunk_with_indicators['slow_ema']).sum()
                chunk_summary['ema_crossovers'] = ema_cross

            if 'vwap' in chunk_with_indicators.columns:
                vwap_bull = (chunk_with_indicators['close'] > chunk_with_indicators['vwap']).sum()
                chunk_summary['vwap_bullish'] = vwap_bull

            chunk_summaries.append(chunk_summary)
            processed_chunks.append(chunk_with_indicators)

            logger.info(f"Chunk {chunk_num} summary: {chunk_summary}")

        except Exception as e:
            logger.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")
            # Use strategy.calculate_indicators as fallback instead of raw data
            fallback_indicators = strategy.calculate_indicators(chunk_df)
            processed_chunks.append(fallback_indicators)

    df_with_indicators = pd.concat(processed_chunks, axis=0, ignore_index=False)

    if len(df_with_indicators) != total_rows:
        logger.error(f"Data integrity check failed: expected {total_rows}, got {len(df_with_indicators)}")
        logger.warning("Falling back to full dataset processing")
        return strategy.calculate_indicators(df_normalized)

    strategy_config = strategy.config.get('strategy', {})
    expected_indicators = []
    if strategy_config.get('use_ema_crossover', False):
        expected_indicators.extend(['fast_ema', 'slow_ema'])
    if strategy_config.get('use_macd', False):
        expected_indicators.extend(['macd', 'macd_signal', 'histogram'])
    if strategy_config.get('use_vwap', False):
        expected_indicators.append('vwap')
    if strategy_config.get('use_atr', False):
        expected_indicators.append('atr')

    missing_indicators = [ind for ind in expected_indicators if ind not in df_with_indicators.columns]
    if missing_indicators:
        logger.error(f"Missing indicators after sequential processing: {missing_indicators}")
        logger.warning("Falling back to full dataset processing")
        return strategy.calculate_indicators(df_normalized)

    logger.info(f"Sequential chunk processing completed successfully: {len(df_with_indicators)} rows with indicators")
    logger.info("=== CHUNK PROCESSING SUMMARY ===")
    total_ema_signals = sum(c['ema_crossovers'] for c in chunk_summaries)
    total_vwap_signals = sum(c['vwap_bullish'] for c in chunk_summaries)
    logger.info(f"Total EMA bullish signals: {total_ema_signals}")
    logger.info(f"Total VWAP bullish signals: {total_vwap_signals}")
    logger.info(f"Both conditions met estimate: {min(total_ema_signals, total_vwap_signals)}")
    return df_with_indicators

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Refactored Backtest Runner with Simple Data Loading")
    parser.add_argument("--config", default="config/strategy_config.yaml", help="Config YAML path")
    parser.add_argument("--data", required=True, help="Path to historical data CSV/LOG")
    
    args = parser.parse_args()
    
    try:
        trades_df, performance = run_backtest(args.config, args.data)
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}")
        raise

"""
CONFIGURATION PARAMETER NAMING CONVENTION:
- This module uses 'config' for all configuration objects
- Function parameter: run_backtest(config, ...)
- Internal usage: strategy_params = config.get('strategy', {})
- Session params: session_params = config.get('session', {})

INTERFACE COMPATIBILITY:
- get_strategy() function maintains 'params' parameter name for interface consistency
- Strategy classes internally use 'config' but receive 'params' from this factory function

CRITICAL: Do not change 'config' variable naming without updating:
- All config.get() calls throughout this file
- Position manager parameter passing
- Session parameter extraction logic
"""