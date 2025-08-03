# test_data_normalizer.py
"""
Unit tests for the DataNormalizer class demonstrating the "normalize early, standardize everywhere" principle.

These tests show how the new centralized data normalization approach handles various data quality issues
and ensures consistent output format for all downstream components.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from data_normalizer import (
    DataNormalizer, 
    CanonicalOHLCVSchema, 
    DataQualityReport, 
    DataQualityIssue,
    normalize_csv_data,
    validate_data_quality
)

class TestDataNormalizer(unittest.TestCase):
    """Test suite for DataNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = DataNormalizer(strict_mode=True, drop_invalid=True)
        self.schema = CanonicalOHLCVSchema()
        
    def create_valid_ohlcv_data(self, num_rows=100):
        """Create valid OHLCV test data."""
        dates = pd.date_range(start='2024-01-01 09:15:00', periods=num_rows, freq='1min')
        
        # Generate realistic OHLCV data
        base_price = 22000.0
        prices = []
        for i in range(num_rows):
            # Random walk for base price
            base_price += np.random.normal(0, 10)
            
            # Generate OHLC based on base price
            open_price = base_price + np.random.normal(0, 5)
            close_price = open_price + np.random.normal(0, 15)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 5))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 5))
            volume = np.random.randint(1000, 10000)
            
            prices.append({
                'timestamp': dates[i],
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume
            })
        
        return pd.DataFrame(prices).set_index('timestamp')
    
    def create_tick_data(self, num_rows=1000):
        """Create tick data for conversion testing."""
        dates = pd.date_range(start='2024-01-01 09:15:00', periods=num_rows, freq='10s')
        
        base_price = 22000.0
        ticks = []
        for i in range(num_rows):
            base_price += np.random.normal(0, 2)
            ticks.append({
                'timestamp': dates[i],
                'price': round(base_price, 2),
                'volume': np.random.randint(100, 1000)
            })
        
        return pd.DataFrame(ticks).set_index('timestamp')
    
    def test_valid_ohlcv_normalization(self):
        """Test normalization of valid OHLCV data."""
        df = self.create_valid_ohlcv_data(100)
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_valid_ohlcv")
        
        # Check that data structure is correct
        self.assertEqual(list(normalized_df.columns), self.schema.REQUIRED_COLUMNS)
        self.assertEqual(normalized_df.index.name, self.schema.INDEX_NAME)
        
        # Check data types
        for col, expected_dtype in self.schema.COLUMN_DTYPES.items():
            self.assertEqual(normalized_df[col].dtype, expected_dtype)
        
        # Check that all rows were processed
        self.assertEqual(report.rows_processed, 100)
        self.assertEqual(report.rows_dropped, 0)
        self.assertEqual(len(report.errors), 0)
        
        # Check timezone
        self.assertEqual(str(normalized_df.index.tz), 'Asia/Kolkata')
    
    def test_tick_data_conversion(self):
        """Test automatic conversion of tick data to OHLCV."""
        tick_df = self.create_tick_data(1000)
        
        normalized_df, report = self.normalizer.normalize_dataframe(tick_df, "test_tick_conversion")
        
        # Should have converted to OHLCV format
        self.assertEqual(list(normalized_df.columns), self.schema.REQUIRED_COLUMNS)
        
        # Should have fewer rows (aggregated into bars)
        self.assertLess(len(normalized_df), 1000)
        self.assertGreater(len(normalized_df), 0)
        
        # Validate OHLC relationships
        for _, row in normalized_df.iterrows():
            self.assertTrue(self.schema.validate_ohlc_relationship(row))
    
    def test_invalid_price_handling(self):
        """Test handling of invalid price data."""
        df = self.create_valid_ohlcv_data(100)
        
        # Introduce invalid prices
        df.iloc[10, 0] = -50.0  # Negative open price
        df.iloc[20, 1] = 2000000.0  # Price too high
        df.iloc[30, 2] = 0.0  # Zero low price
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_invalid_prices")
        
        # Should have dropped invalid rows
        self.assertEqual(report.rows_processed, 97)  # 100 - 3 invalid
        self.assertEqual(report.rows_dropped, 3)
        self.assertGreater(report.issues_found[DataQualityIssue.INVALID_PRICES], 0)
    
    def test_invalid_volume_handling(self):
        """Test handling of invalid volume data."""
        df = self.create_valid_ohlcv_data(100)
        
        # Introduce invalid volumes
        df.iloc[15, 4] = -1000  # Negative volume
        df.iloc[25, 4] = 200000000  # Volume too high
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_invalid_volumes")
        
        # Should have dropped invalid rows
        self.assertEqual(report.rows_processed, 98)  # 100 - 2 invalid
        self.assertEqual(report.rows_dropped, 2)
        self.assertGreater(report.issues_found[DataQualityIssue.INVALID_VOLUMES], 0)
    
    def test_duplicate_timestamp_handling(self):
        """Test handling of duplicate timestamps."""
        df = self.create_valid_ohlcv_data(100)
        
        # Introduce duplicate timestamps
        duplicate_row = df.iloc[50].copy()
        df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=False)
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_duplicates")
        
        # Should have removed duplicates
        self.assertEqual(len(normalized_df), 100)  # Original 100 rows
        self.assertGreater(report.issues_found[DataQualityIssue.DUPLICATE_TIMESTAMPS], 0)
    
    def test_missing_value_handling(self):
        """Test handling of missing values."""
        df = self.create_valid_ohlcv_data(100)
        
        # Introduce missing values
        df.iloc[10, 1] = np.nan  # Missing high price
        df.iloc[20, 4] = np.nan  # Missing volume
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_missing_values")
        
        # Should have dropped rows with missing values (in strict mode)
        self.assertEqual(report.rows_processed, 98)  # 100 - 2 with missing
        self.assertEqual(report.rows_dropped, 2)
        self.assertGreater(report.issues_found[DataQualityIssue.MISSING_VALUES], 0)
        
        # No missing values should remain
        self.assertEqual(normalized_df.isnull().sum().sum(), 0)
    
    def test_ohlc_relationship_validation(self):
        """Test OHLC price relationship validation."""
        df = self.create_valid_ohlcv_data(100)
        
        # Break OHLC relationships
        df.iloc[10, 1] = df.iloc[10, 0] - 10  # High < Open
        df.iloc[20, 2] = df.iloc[20, 3] + 10  # Low > Close
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_ohlc_validation")
        
        # Should have dropped invalid rows
        self.assertEqual(report.rows_processed, 98)  # 100 - 2 invalid
        self.assertEqual(report.rows_dropped, 2)
        
        # Remaining data should have valid OHLC relationships
        for _, row in normalized_df.iterrows():
            self.assertTrue(self.schema.validate_ohlc_relationship(row))
    
    def test_data_type_enforcement(self):
        """Test data type enforcement."""
        df = self.create_valid_ohlcv_data(100)
        
        # Convert to different types
        df['volume'] = df['volume'].astype(float)  # Should be int
        df['close'] = df['close'].astype(str)  # Should be float
        
        # This should fail in the original system but be fixed by normalizer
        try:
            df['close'] = df['close'].astype(float)  # Convert back for testing
        except:
            pass
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_data_types")
        
        # Check that types are enforced
        for col, expected_dtype in self.schema.COLUMN_DTYPES.items():
            self.assertEqual(normalized_df[col].dtype, expected_dtype)
    
    def test_precision_standardization(self):
        """Test price and volume precision standardization."""
        df = self.create_valid_ohlcv_data(100)
        
        # Add excessive precision
        df['close'] = df['close'] + 0.12345  # More than 2 decimal places
        df['volume'] = df['volume'] + 0.789  # Should be integers
        
        normalized_df, report = self.normalizer.normalize_dataframe(df, "test_precision")
        
        # Check precision is standardized
        for _, row in normalized_df.iterrows():
            # Prices should have 2 decimal places max
            self.assertEqual(len(str(row['close']).split('.')[-1]), 2)
            
            # Volume should be integer
            self.assertEqual(row['volume'], int(row['volume']))
    
    def test_strict_mode_error_handling(self):
        """Test error handling in strict mode."""
        df = self.create_valid_ohlcv_data(10)
        
        # Create critical data issue
        df['open'] = -1000  # All negative prices
        
        strict_normalizer = DataNormalizer(strict_mode=True, drop_invalid=True)
        
        # Should raise exception in strict mode
        with self.assertRaises(ValueError):
            strict_normalizer.normalize_dataframe(df, "test_strict_mode")
    
    def test_non_strict_mode_handling(self):
        """Test handling in non-strict mode."""
        df = self.create_valid_ohlcv_data(10)
        
        # Create critical data issue
        df['open'] = -1000  # All negative prices
        
        lenient_normalizer = DataNormalizer(strict_mode=False, drop_invalid=True)
        
        # Should not raise exception in non-strict mode
        normalized_df, report = lenient_normalizer.normalize_dataframe(df, "test_non_strict")
        
        # But should report issues
        self.assertGreater(len(report.errors), 0)
        self.assertTrue(report.has_critical_issues())

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_normalize_csv_data(self):
        """Test CSV file normalization convenience function."""
        # Create temporary CSV file
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:15:00', periods=10, freq='1min'),
            'open': np.random.rand(10) * 100 + 22000,
            'high': np.random.rand(10) * 100 + 22050,
            'low': np.random.rand(10) * 100 + 21950,
            'close': np.random.rand(10) * 100 + 22000,
            'volume': np.random.randint(1000, 5000, 10)
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            normalized_df, report = normalize_csv_data(temp_file, strict_mode=True)
            
            self.assertEqual(len(normalized_df), 10)
            self.assertEqual(list(normalized_df.columns), ['open', 'high', 'low', 'close', 'volume'])
            
        finally:
            os.unlink(temp_file)
    
    def test_validate_data_quality(self):
        """Test data quality validation function."""
        # Create data with known issues
        df = pd.DataFrame({
            'open': [22000, -50, 22010],  # One negative price
            'high': [22050, 22000, 22020],
            'low': [21950, 21950, 21990],
            'close': [22010, np.nan, 22005],  # One missing value
            'volume': [1000, 2000, 3000]
        }, index=pd.date_range('2024-01-01 09:15:00', periods=3, freq='1min'))
        df.index.name = 'timestamp'
        
        report = validate_data_quality(df)
        
        self.assertGreater(report.issues_found[DataQualityIssue.INVALID_PRICES], 0)
        self.assertGreater(report.issues_found[DataQualityIssue.MISSING_VALUES], 0)

class TestIntegrationWithBacktest(unittest.TestCase):
    """Integration tests showing how the normalizer works with backtest components."""
    
    def test_integration_with_strategy(self):
        """Test that normalized data works seamlessly with strategy components."""
        # Create normalized data
        normalizer = DataNormalizer(strict_mode=True, drop_invalid=True)
        raw_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:15:00', periods=100, freq='1min'),
            'open': np.random.rand(100) * 10 + 22000,
            'high': np.random.rand(100) * 10 + 22010,
            'low': np.random.rand(100) * 10 + 21990,
            'close': np.random.rand(100) * 10 + 22000,
            'volume': np.random.randint(1000, 5000, 100)
        }).set_index('timestamp')
        
        normalized_df, report = normalizer.normalize_dataframe(raw_df, "integration_test")
        
        # Test that downstream components can use this data without validation
        # (This simulates what happens in indicators.py and strategy.py)
        
        # 1. Test indicator calculation (assumes clean data)
        normalized_df['sma_20'] = normalized_df['close'].rolling(20).mean()
        self.assertFalse(normalized_df['sma_20'].iloc[-1] != normalized_df['sma_20'].iloc[-1])  # Not NaN
        
        # 2. Test strategy logic (assumes consistent data types)
        for _, row in normalized_df.iterrows():
            self.assertIsInstance(row['close'], (int, float))
            self.assertIsInstance(row['volume'], (int, np.integer))
            self.assertTrue(row['close'] > 0)
            self.assertTrue(row['volume'] >= 0)
        
        # 3. Test that all required columns are present
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertTrue(all(col in normalized_df.columns for col in required_columns))

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)