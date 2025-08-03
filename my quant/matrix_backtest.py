import itertools, os, pandas as pd
from copy import deepcopy
from core.researchStrategy import ModularIntradayStrategy

# Correct imports - only import from backtest_runner.py
from backtest.backtest_runner import run_backtest, get_strategy, load_and_normalize_data

# --- Parameter grid
SL_POINTS = [7, 12]
TA_POINTS = [5, 7, 10]
TD_POINTS = [5, 7, 12]

# --- Indicator-only config (turn all others off)
base_cfg = {
    'strategy': {
        'strategy_version': 'research',
        'use_ema_crossover': True,
        'use_vwap': True,
        'use_macd': False,
        'use_rsi_filter': False,
        'use_htf_trend': False,
        'use_bollinger_bands': False,
        'use_stochastic': False,
        'fast_ema': 9,
        'slow_ema': 21,
        'ema_points_threshold': 0
    },
    'risk': {
        'use_trail_stop': True,
        'tp_points': [10, 25, 50, 100],
        'tp_percents': [0.25, 0.25, 0.25, 0.25],
        'risk_per_trade_percent': 1.0,
        'commission_percent': 0.1,
    },
    'capital': {'initial_capital': 100000},
    'session': {
        'intraday_start_hour': 9,
        'intraday_start_min': 15,
        'intraday_end_hour': 15,
        'intraday_end_min': 30,  # ✅ CORRECTED to match NSE hours
        'exit_before_close': 20,  # User configurable
        'timezone': 'Asia/Kolkata'
    },
    'instrument': {'symbol': 'NIFTY', 'exchange': 'NSE_FO',
                   'lot_size': 50, 'tick_size': 0.05, 'product_type': 'INTRADAY'}
}

def make_cfg(base, sl, ta, td):
    cfg = deepcopy(base)
    cfg['risk']['base_sl_points'] = sl
    cfg['risk']['trail_activation_points'] = ta
    cfg['risk']['trail_distance_points'] = td
    return cfg

def main(data_file, output_dir="matrix_results"):
    os.makedirs(output_dir, exist_ok=True)
    
    # ✅ Use standardized data loading from backtest_runner
    df_normalized, quality_report = load_and_normalize_data(data_file, process_as_ticks=True)
    
    # You can now also log quality metrics:
    print(f"📊 Data loaded: {len(df_normalized)} rows")
    print(f"📊 Data quality metrics: {getattr(quality_report, 'issues_found', {})}")
    
    summary_rows = []
    for sl, ta, td in itertools.product(SL_POINTS, TA_POINTS, TD_POINTS):
        # Build configuration with corrected session parameters
        config = make_cfg(base_cfg, sl, ta, td)
        tag = f"SL{sl}_TA{ta}_TD{td}"
        
        # ✅ Pass normalized data to avoid reloading
        trades, perf = run_backtest(config, data_file, df_normalized=df_normalized)
        
        # ✅ Aggregate results - pure orchestration
        trades.to_csv(f"{output_dir}/{tag}_trades.csv", index=False)
        perf["id"] = tag
        summary_rows.append(perf)
        print(f"✔ Finished {tag}: Trades={perf.get('total_trades', 0)}, P&L={perf.get('total_pnl', 0):.2f}")
    
    pd.DataFrame(summary_rows).to_csv(f"{output_dir}/ema_vwap_matrix.csv", index=False)

if __name__ == "__main__":
    import sys
    # Use command line argument for data file if provided, otherwise use default
    data_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data.csv"
    main(data_file)


"""
CONFIGURATION PARAMETER NAMING CONVENTION:
- Uses 'config' for configuration dictionaries (was 'cfg')
- Calls run_backtest(config, data_file, df_normalized=df_normalized)
- Uses make_config() function that returns 'config' objects

INTERFACE COMPATIBILITY:
- Must match backtest_runner.py parameter naming
- Configuration building follows same structure as main config files

CRITICAL: Keep parameter naming consistent with backtest_runner.py
"""
