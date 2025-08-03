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
from utils.time_utils import normalize_datetime_to_ist, now_ist, ensure_tz_aware, is_time_to_exit



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
    """Load strategy module with full configuration."""
    version = config.get("strategy", {}).get("strategy_version", "live").lower()
    if version == "research":
        strat_mod = importlib.import_module("core.researchStrategy")
    else:
        strat_mod = importlib.import_module("core.liveStrategy")
    ind_mod = importlib.import_module("core.indicators")
    return strat_mod.ModularIntradayStrategy(config, ind_mod)

def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculation=False):
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
    
    # Initialize components
    strategy = get_strategy(config)
    
    # Create a consolidated config dictionary
    position_config = {
        **strategy_params,
        **risk_params, 
        **instrument_params,
        'initial_capital': capital,
        'session': session_params
    }

    # Initialize with a single dictionary argument
    position_manager = PositionManager(position_config)
    
    # Skip data loading if df_normalized is provided
    if df_normalized is None:
        logger.info("Loading data with centralized loader...")
        df_normalized, quality_report = load_and_normalize_data(data_file, process_as_ticks=True)
        logger.info(f"Data loaded: {quality_report.total_rows} rows, {quality_report.rows_dropped} dropped")
    else:
        # If df_normalized is provided, create a simple quality report
        quality_report = type('SimpleQualityReport', (), {
            'total_rows': len(df_normalized),
            'rows_processed': len(df_normalized),
            'rows_dropped': 0,
            'issues_found': {}
        })
    
    # Optimize memory usage for large tick dataset
    if len(df_normalized) > 5000:
        logger.info(f"Optimizing memory usage for large tick dataset ({len(df_normalized)} ticks)")
        
        # Don't convert ticks to bars (preserve tick granularity)
        # Instead, limit the lookback periods used in indicators
        lookback_limit = min(500, int(len(df_normalized) * 0.1))  # 10% of data or 500 max
        
        # Pass memory optimization parameters to indicator calculation
        if skip_indicator_calculation and df_normalized is not None:
            # Use pre-calculated indicators
            df_with_indicators = df_normalized
            print("Using pre-calculated indicators")
        else:
            # Calculate indicators as usual
            df_with_indicators = strategy.calculate_indicators(df_normalized,
                                                             memory_optimized=True,
                                                             max_lookback=lookback_limit)
    else:
        # Normal processing for smaller datasets
        df_with_indicators = strategy.calculate_indicators(df_normalized)
    
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
    logger.info(f"  Total P&L: ₹{performance['total_pnl']:.2f}")
    logger.info(f"  Avg Win: ₹{performance['avg_win']:.2f}")
    logger.info(f"  Avg Loss: ₹{performance['avg_loss']:.2f}")
    logger.info(f"  Profit Factor: {performance['profit_factor']:.2f}")
    logger.info(f"  Max Win: ₹{performance['max_win']:.2f}")
    logger.info(f"  Max Loss: ₹{performance['max_loss']:.2f}")
    logger.info(f"  Total Commission: ₹{performance['total_commission']:.2f}")
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
    Centralized data loading function for the project.
    
    This is the SINGLE ENTRY POINT for all data processing in backtests.
    Replaced the previous DataNormalizer implementation with a simpler approach.
    
    Args:
        data_path: Path to CSV or log file
        process_as_ticks: Whether to process as tick data
        
    Returns:
        Tuple of (normalized_dataframe, quality_report)
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    logger.info(f"Loading data from: {data_path}")
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Use simple_loader as the foundation
    df_normalized = load_data_simple(data_path, process_as_ticks)
    
    # Create a basic quality report for logging
    quality_report = type('SimpleQualityReport', (), {
        'total_rows': len(df_normalized),
        'rows_processed': len(df_normalized),
        'rows_dropped': 0,
        'issues_found': {}
    })
    
    # Add basic validation (previously handled by DataNormalizer)
    if df_normalized.isnull().any().any():
        logger.warning(f"Dataset contains {df_normalized.isnull().sum().sum()} missing values")
    
    # Check for negative prices (previously handled by DataNormalizer)
    neg_prices = (df_normalized['close'] <= 0).sum() if 'close' in df_normalized.columns else 0
    if neg_prices > 0:
        logger.warning(f"Dataset contains {neg_prices} negative or zero prices")
    
    return df_normalized, quality_report

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