from core.liveStrategy import ModularIntradayStrategy
import core.indicators as indicators

# 1. Load or define a full config
config = {
    'use_ema_crossover': True, 'fast_ema': 9, 'slow_ema': 21, 'ema_points_threshold': 2,
    'use_macd': True, 'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
    'use_vwap': True, 'use_htf_trend': True, 'htf_period': 20, 'symbol': 'NIFTY24DECFUT', 'lot_size': 15, 'tick_size': 0.05,
    'session': {'intraday_start_hour': 9, 'intraday_start_min': 15, 'intraday_end_hour': 15, 'intraday_end_min': 30, 'exit_before_close': 20},
    'max_trades_per_day': 25
}

# 2. Pass full config to strategy
strategy = ModularIntradayStrategy(config, indicators)

# 3. Prepare dummy data
import pandas as pd
data = pd.DataFrame({
    'close': [100, 101, 102, 103, 104],
    'volume': [1000, 1100, 1200, 1300, 1400],
    'high': [101, 102, 103, 104, 105],
    'low': [99, 100, 101, 102, 103]
})

# 4. Call indicator calculation
result = strategy.calculate_indicators(data)
print(result.head())