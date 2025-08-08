

# Create the precise diff for -

Remove the manual indicator calculation logic
Replace it with a direct call to calculate_all_indicators()

Based on the conversation history and the attached code files, here's the precise diff to remove the manual indicator calculation logic and replace it with a direct call to `calculate_all_indicators()` in the `process_indicators_sequential` function:

```diff
--- a/backtest_runner.py
+++ b/backtest_runner.py
@@ -457,65 +457,13 @@ def process_indicators_sequential(df_normalized, strategy, chunk_size):
     
     for chunk_num in range(1, num_chunks + 1):
         start_idx = (chunk_num - 1) * chunk_size
         end_idx = min(start_idx + chunk_size, total_rows)
         chunk_df = df_normalized.iloc[start_idx:end_idx].copy()
         
         logger.info(f"Processing chunk {chunk_num}: rows {start_idx}-{end_idx}")
         
         try:
-            # BROKEN: This was using wrong config structure
-            config = strategy.config
-            if config.get('use_ema_crossover', False):
-                # Old code that expected flat config access
-                indicator_states['fast_ema'] = IncrementalEMA(period=config.get('fast_ema', 9))
-                indicator_states['slow_ema'] = IncrementalEMA(period=config.get('slow_ema', 21))
-            
-            # Manual indicator calculation (was broken)
-            for idx, row in chunk_df.iterrows():
-                # EMA indicators
-                if 'fast_ema' in indicator_states:
-                    chunk_df.loc[idx, 'fast_ema'] = indicator_states['fast_ema'].update(row['close'])
-                    chunk_df.loc[idx, 'slow_ema'] = indicator_states['slow_ema'].update(row['close'])
-                
-                # MACD indicator
-                if 'macd' in indicator_states:
-                    macd_val, signal_val, hist_val = indicator_states['macd'].update(row['close'])
-                    chunk_df.loc[idx, 'macd'] = macd_val
-                    chunk_df.loc[idx, 'macd_signal'] = signal_val
-                    chunk_df.loc[idx, 'histogram'] = hist_val
-                
-                # VWAP indicator
-                if 'vwap' in indicator_states:
-                    vwap_val = indicator_states['vwap'].update(
-                        price=row['close'],
-                        volume=row['volume'],
-                        high=row.get('high'),
-                        low=row.get('low'),
-                        close=row.get('close')
-                    )
-                    chunk_df.loc[idx, 'vwap'] = vwap_val
-                
-                # ATR indicator
-                if 'atr' in indicator_states:
-                    atr_val = indicator_states['atr'].update(
-                        high=row['high'],
-                        low=row['low'],
-                        close=row['close']
-                    )
-                    chunk_df.loc[idx, 'atr'] = atr_val
-            
-            # Add signal calculations based on computed indicators
-            add_indicator_signals_to_chunk(chunk_df, config)
+            # FIXED: Pass nested config directly to calculate_all_indicators
+            chunk_with_indicators = calculate_all_indicators(chunk_df, strategy.config)
             
             # Collect diagnostic info for this chunk
             chunk_summary = {
                 'chunk': chunk_num,
-                'rows': len(chunk_df),
-                'time_start': chunk_df.index[^0],
-                'time_end': chunk_df.index[-1],
+                'rows': len(chunk_with_indicators),
+                'time_start': chunk_with_indicators.index[^0],
+                'time_end': chunk_with_indicators.index[-1],
                 'ema_crossovers': 0,
                 'vwap_bullish': 0
             }
             
-            if 'fast_ema' in chunk_df.columns and 'slow_ema' in chunk_df.columns:
-                ema_cross = (chunk_df['fast_ema'] > chunk_df['slow_ema']).sum()
+            if 'fast_ema' in chunk_with_indicators.columns and 'slow_ema' in chunk_with_indicators.columns:
+                ema_cross = (chunk_with_indicators['fast_ema'] > chunk_with_indicators['slow_ema']).sum()
                 chunk_summary['ema_crossovers'] = ema_cross
             
-            if 'vwap' in chunk_df.columns:
-                vwap_bull = (chunk_df['close'] > chunk_df['vwap']).sum()
+            if 'vwap' in chunk_with_indicators.columns:
+                vwap_bull = (chunk_with_indicators['close'] > chunk_with_indicators['vwap']).sum()
                 chunk_summary['vwap_bullish'] = vwap_bull
             
             chunk_summaries.append(chunk_summary)
-            processed_chunks.append(chunk_df)
+            processed_chunks.append(chunk_with_indicators)
             
             logger.info(f"Chunk {chunk_num} summary: {chunk_summary}")
             
         except Exception as e:
             logger.error(f"Error processing chunk {start_idx}-{end_idx}: {e}")
             # Fallback: use original data for this chunk
             processed_chunks.append(chunk_df)
```


## Key Changes Made:

1. **Removed manual indicator initialization**: Eliminated the complex `indicator_states` dictionary and individual `IncrementalEMA`, `IncrementalMACD`, etc. trackers
2. **Removed row-by-row processing**: Eliminated the entire `for idx, row in chunk_df.iterrows():` loop that was manually calculating indicators
3. **Replaced with direct call**: Added `chunk_with_indicators = calculate_all_indicators(chunk_df, strategy.config)` which handles all indicator calculations using the proper nested config structure
4. **Updated variable references**: Changed `chunk_df` to `chunk_with_indicators` in the diagnostic and summary sections to use the properly calculated data
5. **Removed helper function call**: Eliminated `add_indicator_signals_to_chunk(chunk_df, config)` as the signal calculations are now handled within `calculate_all_indicators()`

This change simplifies the code significantly while fixing the configuration access issues and ensuring that all indicators are properly calculated using the nested config structure that was implemented in the previous fixes.

<div style="text-align: center">⁂</div>

[^1]: config_helper.py

[^2]: strategy_config.yaml

[^3]: indicators.py

[^4]: liveStrategy.py

[^5]: position_manager.py

[^6]: researchStrategy.py

[^7]: unified_gui.py

[^8]: login.py

[^9]: cache_manager.py

[^10]: config_helper.py

[^11]: config_loader.py

[^12]: logging_utils.py

[^13]: simple_loader.py

[^14]: time_utils.py

[^15]: researchStrategy.py

[^16]: backtest_runner.py

