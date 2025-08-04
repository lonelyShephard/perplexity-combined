"""
backtest/backtest_runner.py

Unified backtest engine integrating live and research strategies,
full parameterization, and modular position/risk/trade management
for long-only, intraday, F&O-ready workflows.
"""

import yaml
import importlib
import logging
import pandas as pd
from datetime import datetime

from core.position_manager import PositionManager

try:
    from core.utils import ensure_tz_aware, is_time_to_exit
    logger.info("✅ Timezone utilities loaded")
except ImportError as e:
    logger.error(f"❌ Failed to load timezone utilities: {e}")
    raise

def load_config(config_path: str) -> dict:
    """Load YAML strategy and risk config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_strategy(config: dict):
    """Load strategy module with full configuration.
    
    Args:
        config: Complete configuration dictionary
    """
    version = config.get("strategy", {}).get("strategy_version", "live").lower()
    if version == "research":
        strat_mod = importlib.import_module("core.researchStrategy")
    else:
        strat_mod = importlib.import_module("core.liveStrategy")
    
    ind_mod = importlib.import_module("core.indicators")
    return strat_mod.ModularIntradayStrategy(config, ind_mod)

def load_data(data_path: str) -> pd.DataFrame:
    """Load historical OHLCV data from CSV file, sorts and indexes by timestamp."""
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df

def run_backtest(config, data_file, df_normalized=None, skip_indicator_calculation=False):
    """Run a backtest with the given configuration.
    
    Args:
        config: Configuration dictionary or path to config file
        data_file: Path to data file (CSV/LOG)
        df_normalized: Pre-normalized data (optional)
        skip_indicator_calculation: Skip indicator calculation if data already has them
    """
    logger.info("=" * 60)
    logger.info("STARTING BACKTEST WITH NORMALIZED DATA PIPELINE")
    logger.info("=" * 60)
    
    # Load configuration
    if isinstance(config, str):
        config = load_config(config)
    
    # Extract parameters with validation
    try:
        strategy_params = config.get('strategy', {})
        session_params = config.get('session', {})
        risk_params = config.get('risk', {})
        instrument_params = config.get('instrument', {})
        capital = config.get('capital', {}).get('initial_capital', 100000)
        if capital <= 0:
            raise ValueError("Initial capital must be positive")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    # Initialize components
    strategy = get_strategy(config)
    
    # Create consolidated config for position manager
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
        logger.info("Loading historical data for backtest...")
        df_normalized = load_data(data_file)
        df_ind = strategy.calculate_indicators(df_normalized)
    else:
        df_ind = df_normalized
    
    # Validate data quality
    if df_normalized.empty:
        raise ValueError("No data available for backtest")
    logger.info(f"Starting backtest with {len(df_normalized)} data points")
    
    # Optimize memory usage for large tick dataset
    if len(df_normalized) > 5000:
        logger.info(f"Optimizing memory usage for large tick dataset ({len(df_normalized)} ticks)")
        lookback_limit = min(500, int(len(df_normalized) * 0.1))  # 10% of data or 500 max
        
        # Pass memory optimization parameters to indicator calculation
        if skip_indicator_calculation:
            # Use pre-calculated indicators
            df_with_indicators = df_normalized
            logger.info("Using pre-calculated indicators")
        else:
            # Calculate indicators as usual
            df_with_indicators = strategy.calculate_indicators(
                df_normalized,
                memory_optimized=True,
                max_lookback=lookback_limit)
    else:
        df_with_indicators = df_ind
    
    logger.info("Starting backtest execution...")
    position_id = None
    in_position = False
    processed_bars = 0
    
    # Extract session parameters for exit logic
    close_hour = session_params.get("intraday_end_hour", 15)
    close_min = session_params.get("intraday_end_min", 30)
    exit_buffer = session_params.get("exit_before_close", 20)
    
    if close_hour < 0 or close_hour > 23:
        logger.warning(f"Invalid close hour {close_hour}, using default 15")
        close_hour = 15
    if close_min < 0 or close_min > 59:
        logger.warning(f"Invalid close minute {close_min}, using default 30")
        close_min = 30
    
    for timestamp, row in df_with_indicators.iterrows():
        processed_bars += 1
        # ENSURE timezone awareness for timestamp
        now = ensure_tz_aware(timestamp)
        # Add session exit flag to row
        row['session_exit'] = is_time_to_exit(now, exit_buffer, close_hour, close_min)
        # Check if in exit buffer period - CORE LOGIC
        if is_time_to_exit(now, exit_buffer, close_hour, close_min):
            # Close all positions and terminate
            for pos_id in list(position_manager.positions.keys()):
                position_manager.close_position_full(pos_id, row['close'], now, "Exit Buffer")
            logger.info("Exit buffer reached - all positions closed")
            break  # Stop processing completely
        
        # For debugging the first few iterations
        if processed_bars <= 3:
            logger.info(f"Processing timestamp: {now} (tzinfo: {now.tzinfo})")
            logger.info(f"Row data: close={row.get('close', 'N/A')}, volume={row.get('volume', 'N/A')}")
        if pd.isna(row.get('close')):
            logger.warning(f"Invalid close price at {now}, skipping")
            continue
        position_manager.process_positions(row, now, session_params)
        # Entry Logic: only if not already in position and conditions are met
        if not in_position and strategy.can_open_long(row, now):
            position_id = strategy.open_long(row, now, position_manager)
            in_position = position_id is not None
            if in_position:
                logger.debug(f"Opened position {position_id} at {now} @ {row['close']:.2f}")
        # Exit Logic: PositionManager handles trailing stops, TPs, SLs
        if in_position:
            # Check for strategy-level exit conditions
            if strategy.should_close(row, now, position_manager):
                last_price = row['close']
                strategy.handle_exit(position_id, last_price, now, position_manager, reason="Strategy Exit")
                in_position = False
                position_id = None
                logger.debug(f"Strategy exit at {now} @ {last_price:.2f}")
        # Reset position state if position closed by PositionManager
        if position_id and position_id not in position_manager.positions:
            in_position = False
            position_id = None
        # Log first timestamp info
        if processed_bars <= 2:
            logger.info(f"First timestamp processing details:")
            logger.info(f" - Original timestamp: {timestamp} (tzinfo: {timestamp.tzinfo})")
            logger.info(f" - Normalized 'now': {now} (tzinfo: {now.tzinfo})")
            logger.info(f" - Session exit check: {row['session_exit']}")
            logger.info(f" - Session live check: {strategy.is_session_live(now)}")
    # Defensive: flatten any still-open positions at backtest end
    if position_id and position_id in position_manager.positions:
        try:
            last_price = df_with_indicators.iloc[-1]['close']
        except (IndexError, KeyError):
            last_price = 0.0
        now = df_with_indicators.index[-1]
        strategy.handle_exit(position_id, last_price, now, position_manager, reason="End of Backtest")
        logger.info(f"Closed final position at backtest end @ {last_price:.2f}")
    
    # Gather and print summary
    trades = position_manager.get_trade_history()
    performance = position_manager.get_performance_summary()

    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Data Quality:")
    total_rows = getattr(quality_report, 'total_rows', len(df_normalized))
    processed_rows = getattr(quality_report, 'rows_processed', len(df_normalized))
    logger.info(f" Total input rows: {total_rows}")
    logger.info(f" Processed rows: {processed_rows}")
    logger.info(f" Data quality: {quality_report.rows_processed/quality_report.total_rows*100:.1f}%")
    logger.info("")
    logger.info(f"Trading Performance:")
    logger.info(f" Total Trades: {performance['total_trades']}")
    logger.info(f" Win Rate   : {performance['win_rate']:.2f}%")
    logger.info(f" Total P&L  : ₹{performance['total_pnl']:.2f}")
    logger.info(f" Avg Win    : ₹{performance['avg_win']:.2f}")
    logger.info(f" Avg Loss   : ₹{performance['avg_loss']:.2f}")
    logger.info(f" ProfitFactor: {performance['profit_factor']:.2f}")
    logger.info(f" Max Win    : ₹{performance['max_win']:.2f}")
    logger.info(f" Max Loss   : ₹{performance['max_loss']:.2f}")
    logger.info(f" Total Commission: ₹{performance['total_commission']:.2f}")
    logger.info("=" * 60)
    # Save trade log CSV file
    try:
        if trades and len(trades) > 0:
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("backtest_trades.csv", index=False)
            logger.info("Trade log written to backtest_trades.csv")
        else:
            logger.warning("No trades executed during backtest")
            trades_df = pd.DataFrame()
    except Exception as e:
        logger.error(f"Error saving trade log: {e}")
        trades_df = pd.DataFrame(trades)
    return trades_df, performance

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')
    parser = argparse.ArgumentParser(description="Unified Backtest Runner")
    parser.add_argument("--config", default="config/strategy_config.yaml", help="Config YAML path")
    parser.add_argument("--data", required=True, help="Path to historical data CSV")
    args = parser.parse_args()

    run_backtest(args.config, args.data)
