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

def load_config(config_path: str) -> dict:
    """Load YAML strategy and risk config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_strategy(params: dict):
    """Dynamically load strategy module based on config parameter `strategy_version`."""
    version = params.get("strategy_version", "live").lower()
    if version == "research":
        strat_mod = importlib.import_module("core.researchStrategy")
    else:
        strat_mod = importlib.import_module("core.liveStrategy")
    ind_mod = importlib.import_module("core.indicators")
    return strat_mod.ModularIntradayStrategy(params, ind_mod)

def load_data(data_path: str) -> pd.DataFrame:
    """Load historical OHLCV data from CSV file, sorts and indexes by timestamp."""
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df

def run_backtest(config_source, data_path: str):
    """
    Run the backtest engine with unified strategy and position manager.

    Args:
        config_source: YAML config file path (str) or config dictionary (dict)
        data_path: CSV historical data path

    Returns:
        trades_df: DataFrame of executed trades
        performance: Performance summary dict
    """
    # Load configuration and parameters
    if isinstance(config_source, str):
        config = load_config(config_source)
    elif isinstance(config_source, dict):
        config = config_source
    else:
        raise ValueError("config_source must be either a file path (str) or config dictionary (dict)")
        
    strategy_params = config['strategy']
    risk_params = config.get('risk', {})
    session_params = config.get('session', {})
    instrument_params = config.get('instrument', {})
    capital = config.get('capital', {}).get('initial_capital', 100000)

    # Initialize strategy and position manager instances
    strategy = get_strategy(strategy_params)
    position_manager = PositionManager({
        **strategy_params,
        **risk_params,
        **instrument_params,
        "initial_capital": capital,
        "session": session_params,
    })

    logging.info("Loading historical data for backtest...")
    df = load_data(data_path)
    df_ind = strategy.calculate_indicators(df)

    position_id = None
    in_position = False

    session_end_hour = session_params.get("intraday_end_hour", 15)
    session_end_min = session_params.get("intraday_end_min", 15)
    exit_before_close = session_params.get("exit_before_close", 20)

    for timestamp, row in df_ind.iterrows():
        now = timestamp

        # Determine if we should exit due to session close buffer
        session_end_dt = datetime.combine(now.date(), datetime.min.time()).replace(
            hour=session_end_hour, minute=session_end_min)
        row['session_exit'] = now >= (session_end_dt - pd.Timedelta(minutes=exit_before_close))

        # Entry Logic: only if not already in position and conditions meet
        if not in_position and strategy.can_open_long(row, now):
            position_id = strategy.open_long(row, now, position_manager)
            in_position = position_id is not None

        # Exit Logic: PositionManager handles trailing stops, TPs, SLs and session-end exits
        if in_position:
            position_manager.process_positions(row, now)

            if strategy.should_close(row, now, position_manager):
                last_price = row['close']
                strategy.handle_exit(position_id, last_price, now, position_manager, reason="Session End")
                in_position = False
                position_id = None
        else:
            # Still allow PositionManager to process open positions in edge cases
            position_manager.process_positions(row, now)

        # Reset position state if position closed by PositionManager
        if position_id and position_id not in position_manager.positions:
            in_position = False
            position_id = None

    # Defensive: flatten any still-open positions at backtest end
    if position_id and position_id in position_manager.positions:
        last_price = df_ind.iloc[-1]['close']
        now = df_ind.index[-1]
        strategy.handle_exit(position_id, last_price, now, position_manager, reason="End of Backtest")

    # Gather and print summary
    trades = position_manager.get_trade_history()
    performance = position_manager.get_performance_summary()

    print("\n---- BACKTEST SUMMARY ----")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Win Rate   : {performance['win_rate']:.2f}%")
    print(f"Total P&L  : ₹{performance['total_pnl']:.2f}")
    print(f"Avg Win    : ₹{performance['avg_win']:.2f}")
    print(f"Avg Loss   : ₹{performance['avg_loss']:.2f}")
    print(f"ProfitFactor: {performance['profit_factor']:.2f}")
    print(f"Max Win    : ₹{performance['max_win']:.2f}")
    print(f"Max Loss   : ₹{performance['max_loss']:.2f}")
    print(f"Total Commission: ₹{performance['total_commission']:.2f}")

    # Save trade log CSV file
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv("backtest_trades.csv", index=False)
    print("\nTrade log written to backtest_trades.csv")

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
