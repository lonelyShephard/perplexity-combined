"""
core/indicators.py
Unified, parameter-driven indicator library for both backtest and live trading bot.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

def safe_divide(numerator, denominator, default=0.0):
    return numerator / denominator if denominator != 0 else default

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

def calculate_ema_crossover_signals(fast_ema: pd.Series, slow_ema: pd.Series, threshold: float = 0) -> pd.DataFrame:
    crossover = fast_ema > (slow_ema + threshold)
    prev = crossover.shift(1)
    return pd.DataFrame({
        'bullish_cross': crossover & (~prev),
        'bearish_cross': (~crossover) & prev,
        'ema_bullish': crossover
    })

def calculate_macd_signals(macd_df: pd.DataFrame) -> pd.DataFrame:
    above = macd_df['macd'] > macd_df['signal']
    prev = above.shift()
    return pd.DataFrame({
        'macd_buy_signal': above & (~prev),
        'macd_sell_signal': (~above) & prev,
        'macd_bullish': above,
        'macd_histogram_positive': macd_df['histogram'] > 0
    })

def calculate_htf_signals(close: pd.Series, htf_ema: pd.Series) -> pd.DataFrame:
    bullish = close > htf_ema
    return pd.DataFrame({
        'htf_bullish': bullish,
        'htf_bearish': ~bullish
    })

def calculate_vwap_signals(close: pd.Series, vwap: pd.Series) -> pd.DataFrame:
    bullish = close > vwap
    return pd.DataFrame({
        'vwap_bullish': bullish,
        'vwap_bearish': ~bullish
    })

def calculate_rsi_signals(rsi: pd.Series, overbought: float = 70, oversold: float = 30) -> pd.DataFrame:
    return pd.DataFrame({
        'rsi_oversold': rsi <= oversold,
        'rsi_overbought': rsi >= overbought,
        'rsi_neutral': (rsi > oversold) & (rsi < overbought)
    })

def calculate_all_indicators(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    df = df.copy()

    if params.get("use_ema_crossover"):
        df['fast_ema'] = calculate_ema(df['close'], params.get("fast_ema", 9))
        df['slow_ema'] = calculate_ema(df['close'], params.get("slow_ema", 21))
        emacross = calculate_ema_crossover_signals(df['fast_ema'], df['slow_ema'], params.get("ema_points_threshold", 0))
        df = df.join(emacross)

    if params.get("use_macd"):
        macd_df = calculate_macd(df['close'], params.get("macd_fast", 12), params.get("macd_slow", 26), params.get("macd_signal", 9))
        df = df.join(macd_df)
        macd_signals = calculate_macd_signals(macd_df)
        df = df.join(macd_signals)

    if params.get("use_rsi_filter"):
        df['rsi'] = calculate_rsi(df['close'], params.get("rsi_length", 14))
        df = df.join(calculate_rsi_signals(df['rsi'], params.get("rsi_overbought", 70), params.get("rsi_oversold", 30)))

    if params.get("use_vwap") and all(col in df.columns for col in ['high', 'low', 'volume']):
        df['vwap'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
        df = df.join(calculate_vwap_signals(df['close'], df['vwap']))

    if params.get("use_htf_trend"):
        df['htf_ema'] = calculate_htf_trend(df['close'], params.get("htf_period", 20))
        df = df.join(calculate_htf_signals(df['close'], df['htf_ema']))

    if params.get("use_bollinger_bands"):
        upper, mid, lower = calculate_bollinger_bands(df['close'], params.get("bb_period", 20), params.get("bb_std", 2))
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = upper, mid, lower

    if params.get("use_stochastic"):
        k, d = calculate_stochastic(df['high'], df['low'], df['close'], params.get("stoch_k", 14), params.get("stoch_d", 3))
        df["stoch_k"], df["stoch_d"] = k, d

    if params.get("use_ma"):
        df["ma_short"] = calculate_sma(df["close"], params.get("ma_short", 20))
        df["ma_long"] = calculate_sma(df["close"], params.get("ma_long", 50))

    if params.get("use_atr") and all(x in df.columns for x in ['high', 'low']):
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], params.get("atr_len", 14))

    if params.get("use_volume_ma") and "volume" in df.columns:
        df["volume_ma"] = calculate_sma(df["volume"], params.get("volume_ma_period", 20))
        df["volume_ratio"] = safe_divide(df["volume"], df["volume_ma"])

    return df
