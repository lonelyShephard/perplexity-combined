# MIGRATION GUIDE: Implementing "Normalize Early, Standardize Everywhere"

## Overview

This guide walks you through migrating your existing backtest system to implement the "normalize early, standardize everywhere" principle. The refactoring centralizes all data normalization into a single entry point, improving data quality, performance, and maintainability.

## Step-by-Step Migration

### Step 1: Add the New Data Normalizer Module

1. **Copy the new module**: Add `data_normalizer.py` to your project root
2. **Install dependencies**: Ensure you have the required imports
3. **Verify imports**: Make sure `utils.time_utils` is accessible

```bash
# Verify the module can be imported
python -c "from data_normalizer import DataNormalizer; print('✅ Import successful')"
```

### Step 2: Update Your Backtest Runner

**Option A: Replace the existing file**
```bash
# Backup original
cp backtest/backtest_runner.py backtest/backtest_runner_original.py

# Replace with refactored version
cp backtest_runner_refactored.py backtest/backtest_runner.py
```

**Option B: Gradual migration (safer approach)**

1. **Update imports** in your existing `backtest_runner.py`:
```python
# Add at the top of backtest_runner.py
from data_normalizer import DataNormalizer, DataQualityReport
```

2. **Replace the load_data function**:
```python
# Replace the existing load_data function with this:
def load_data(data_path: str, granularity: str = "bars", bar_minutes: int = 1):
    """Load and normalize data using the new centralized approach."""
    
    # Use the new normalized data loader
    from data_normalizer import DataNormalizer
    import pandas as pd
    import os
    
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Simple file loading (let normalizer handle format detection)
    _, ext = os.path.splitext(data_path)
    if ext.lower() == ".log":
        df = pd.read_csv(data_path, names=['timestamp', 'price', 'volume'], 
                        parse_dates=[0], header=None)
    else:
        df = pd.read_csv(data_path, parse_dates=[0], header=0)
    
    # NORMALIZE EARLY - SINGLE SOURCE OF TRUTH
    normalizer = DataNormalizer(strict_mode=True, drop_invalid=True)
    normalized_df, quality_report = normalizer.normalize_dataframe(
        df, f"backtest:{os.path.basename(data_path)}"
    )
    
    # Log data quality results
    logger.info(f"Data loaded: {quality_report.rows_processed} rows processed, "
                f"{quality_report.rows_dropped} rows dropped")
    
    return normalized_df
```

3. **Update the main run_backtest function**:
```python
def run_backtest(config_source, data_path: str):
    # ... existing config loading code ...
    
    logger.info("Loading historical data for backtest...")
    
    # Use the new data loading approach
    df = load_data(data_path)  # Now returns normalized data
    
    # Calculate indicators (no need for additional data validation)
    df_ind = strategy.calculate_indicators(df)
    
    # ... rest of backtest logic remains the same ...
```

### Step 3: Clean Up Downstream Components

Since data is now normalized at entry, you can remove redundant validation code:

**In `core/indicators.py`**:
```python
# REMOVE these kinds of checks (data is pre-normalized):
# if pd.isna(row['close']) or row['close'] <= 0:
#     return False

# KEEP the business logic:
def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()  # Just the calculation
```

**In strategy files**:
```python
# REMOVE data validation checks like:
# if 'close' not in row or pd.isna(row['close']):
#     return False

# KEEP the trading logic:
def should_enter_long(self, row: pd.Series) -> bool:
    # Focus on trading conditions only
    return (row['fast_ema'] > row['slow_ema'] and 
            row['close'] > row['vwap'])
```

### Step 4: Update Configuration (Optional)

Add data quality settings to your `config/strategy_config.yaml`:

```yaml
# Add to your existing config file
data_quality:
  strict_mode: true          # Fail fast on critical data issues
  drop_invalid: true         # Drop bad rows instead of trying to fix
  price_precision: 2         # Decimal places for prices
  volume_precision: 0        # Decimal places for volumes (integers)
  min_price: 0.01           # Minimum valid price
  max_price: 1000000.0      # Maximum valid price
  min_volume: 0             # Minimum valid volume
  max_volume: 100000000     # Maximum valid volume
```

Then update your backtest runner to use these settings:
```python
def run_backtest(config_source, data_path: str):
    config = load_config(config_source) if isinstance(config_source, str) else config_source
    
    # Get data quality settings
    data_quality_config = config.get('data_quality', {})
    strict_mode = data_quality_config.get('strict_mode', True)
    
    # Use in data loading
    normalizer = DataNormalizer(
        strict_mode=strict_mode,
        drop_invalid=data_quality_config.get('drop_invalid', True)
    )
    # ... rest of the code
```

### Step 5: Update Your GUI (If Using)

In `gui/unified_gui.py`, update the backtest execution:

```python
def _bt_run_backtest(self):
    # ... existing GUI code ...
    
    # Update the worker function to use new backtest runner
    def _bt_worker(config_dict, data_path):
        try:
            # Import the updated run_backtest function
            from backtest.backtest_runner import run_backtest
            
            trades_df, metrics = run_backtest(config_dict, data_path)
            
            # Display results including data quality info if available
            summary = f"---- BACKTEST SUMMARY ----\n"
            summary += f"Total Trades: {metrics['total_trades']}\n"
            # ... existing summary code ...
            
            self.bt_result_box.config(state="normal")
            self.bt_result_box.delete("1.0", "end")
            self.bt_result_box.insert("end", summary)
            self.bt_result_box.config(state="disabled")
            
        except Exception as e:
            # Better error handling
            error_msg = f"Backtest failed: {str(e)}\n"
            if "data quality" in str(e).lower():
                error_msg += "\nTip: Check your input data for missing values, invalid prices, or format issues."
                
            self.bt_result_box.config(state="normal")
            self.bt_result_box.insert("end", error_msg)
            self.bt_result_box.config(state="disabled")
```

### Step 6: Testing Your Migration

1. **Run the unit tests**:
```bash
python test_data_normalizer.py -v
```

2. **Test with your existing data**:
```bash
python backtest/backtest_runner.py --data your_test_data.csv --config config/strategy_config.yaml
```

3. **Compare results** with the old system:
```bash
# Run old version (backed up)
python backtest/backtest_runner_original.py --data test_data.csv --config config.yaml > old_results.txt

# Run new version
python backtest/backtest_runner.py --data test_data.csv --config config.yaml > new_results.txt

# Compare (should be very similar, but new version may have fewer trades due to better data quality)
diff old_results.txt new_results.txt
```

### Step 7: Monitoring and Validation

After migration, monitor for these improvements:

**Data Quality Metrics**:
```python
# Add to your backtest logging
logger.info("Data Quality Report:")
logger.info(f"  Data completeness: {(report.rows_processed/report.total_rows)*100:.1f}%")
logger.info(f"  Rows dropped: {report.rows_dropped}")
logger.info(f"  Issues found: {sum(report.issues_found.values())}")
```

**Performance Monitoring**:
```python
import time

# Time the data loading
start_time = time.time()
df_normalized, report = load_and_normalize_data(data_path)
load_time = time.time() - start_time

logger.info(f"Data loading completed in {load_time:.2f} seconds")
logger.info(f"Processing rate: {len(df_normalized)/load_time:.0f} rows/second")
```

## Troubleshooting Common Issues

### Issue 1: Import Errors
```
ImportError: cannot import name 'DataNormalizer' from 'data_normalizer'
```
**Solution**: Ensure `data_normalizer.py` is in your project root and `utils.time_utils` is accessible.

### Issue 2: Data Type Errors
```
ValueError: cannot convert float NaN to integer
```
**Solution**: The normalizer is finding missing values. Check your input data or set `strict_mode=False`.

### Issue 3: All Data Dropped
```
Warning: All rows were dropped during normalization
```
**Solution**: Your data may have critical quality issues. Run with `strict_mode=False` first to see what's wrong:

```python
# Debug data quality issues
normalizer = DataNormalizer(strict_mode=False, drop_invalid=False)
df_normalized, report = normalizer.normalize_dataframe(df, "debug")

print("Issues found:")
for issue, count in report.issues_found.items():
    if count > 0:
        print(f"  {issue.value}: {count}")

print("Errors:")
for error in report.errors:
    print(f"  {error}")
```

### Issue 4: Performance Degradation
If the new system is slower:

1. **Check data quality**: Poor quality data requires more processing
2. **Use appropriate settings**:
```python
# For large datasets, consider:
normalizer = DataNormalizer(
    strict_mode=False,    # Don't fail on every issue
    drop_invalid=True     # But still drop bad rows
)
```

### Issue 5: Different Backtest Results
If results differ significantly:

1. **Check dropped rows**: The normalizer may be removing bad data that was previously processed
2. **Verify precision**: Prices are now standardized to 2 decimal places
3. **Check timestamps**: All timestamps are now properly normalized to IST

## Validation Checklist

Before deploying to production:

- [ ] All unit tests pass
- [ ] Backtest runs successfully with your data
- [ ] Data quality reports show reasonable metrics
- [ ] Performance is acceptable
- [ ] Results are consistent with expectations
- [ ] Error handling works for bad data files
- [ ] Logging provides useful information

## Rollback Plan

If you need to rollback:

1. **Restore original files**:
```bash
cp backtest/backtest_runner_original.py backtest/backtest_runner.py
```

2. **Remove new imports** from any files you modified

3. **Test that original system works**

## Benefits You Should See

After successful migration:

✅ **Consistent data quality** across all backtests  
✅ **Faster debugging** when data issues occur  
✅ **Better error messages** for data problems  
✅ **Cleaner component code** (less validation, more business logic)  
✅ **Easier testing** with predictable data formats  
✅ **Better performance** from single-pass data processing  

The refactoring transforms your data pipeline from ad-hoc transformations into a robust, enterprise-grade system that implements data engineering best practices.