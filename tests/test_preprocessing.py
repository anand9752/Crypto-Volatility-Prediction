import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample cryptocurrency data for testing"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        data = {
            'date': dates,
            'crypto_name': ['Bitcoin'] * len(dates),
            'open': np.random.uniform(20000, 25000, len(dates)),
            'high': np.random.uniform(25000, 30000, len(dates)),
            'low': np.random.uniform(15000, 20000, len(dates)),
            'close': np.random.uniform(20000, 25000, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates)),
            'marketCap': np.random.uniform(400000000000, 500000000000, len(dates))
        }
        
        # Ensure high >= low constraint
        for i in range(len(dates)):
            data['high'][i] = max(data['high'][i], data['low'][i], data['open'][i], data['close'][i])
            data['low'][i] = min(data['low'][i], data['open'][i], data['close'][i])
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance"""
        return DataPreprocessor()
    
    def test_basic_cleaning(self, preprocessor, sample_data):
        """Test basic data cleaning functionality"""
        # Add an unnamed column to test removal
        sample_data['Unnamed: 0'] = range(len(sample_data))
        
        cleaned_data = preprocessor.basic_cleaning(sample_data)
        
        # Check that unnamed column is removed
        assert 'Unnamed: 0' not in cleaned_data.columns
        
        # Check that date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(cleaned_data['date'])
        
        # Check that data is sorted
        assert cleaned_data['date'].is_monotonic_increasing
    
    def test_handle_missing_values(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Introduce missing values
        sample_data.loc[5:7, 'close'] = np.nan
        sample_data.loc[10:12, 'volume'] = np.nan
        
        processed_data = preprocessor.handle_missing_values(sample_data)
        
        # Check that missing values are handled
        assert processed_data['close'].isnull().sum() == 0
        assert processed_data['volume'].isnull().sum() == 0
    
    def test_detect_outliers(self, preprocessor, sample_data):
        """Test outlier detection"""
        # Introduce obvious outliers
        sample_data.loc[5, 'close'] = 1000000  # Extremely high price
        sample_data.loc[10, 'volume'] = 1  # Extremely low volume
        
        data_with_outliers = preprocessor.detect_outliers(sample_data)
        
        # Check that outlier column is added
        assert 'is_outlier' in data_with_outliers.columns
        
        # Check that outliers are detected
        assert data_with_outliers['is_outlier'].sum() > 0
    
    def test_validate_data_quality(self, preprocessor, sample_data):
        """Test data quality validation"""
        quality_report = preprocessor.validate_data_quality(sample_data)
        
        # Check that quality report contains expected keys
        expected_keys = ['total_records', 'total_cryptocurrencies', 'date_range', 
                        'missing_values', 'data_types']
        for key in expected_keys:
            assert key in quality_report
        
        # Check basic statistics
        assert quality_report['total_records'] == len(sample_data)
        assert quality_report['total_cryptocurrencies'] == 1
    
    def test_create_time_features(self, preprocessor, sample_data):
        """Test time-based feature creation"""
        sample_data['date'] = pd.to_datetime(sample_data['date'])
        data_with_time_features = preprocessor.create_time_features(sample_data)
        
        # Check that time features are created
        time_features = ['year', 'month', 'day', 'day_of_week', 'quarter', 
                        'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        
        for feature in time_features:
            assert feature in data_with_time_features.columns
        
        # Check feature values
        assert data_with_time_features['year'].iloc[0] == 2023
        assert data_with_time_features['month'].iloc[0] == 1
    
    def test_filter_by_date_range(self, preprocessor, sample_data):
        """Test date range filtering"""
        filtered_data = preprocessor.filter_by_date_range(
            sample_data, 
            start_date='2023-01-10', 
            end_date='2023-01-20'
        )
        
        # Check that data is filtered correctly
        assert len(filtered_data) <= len(sample_data)
        assert filtered_data['date'].min() >= pd.to_datetime('2023-01-10')
        assert filtered_data['date'].max() <= pd.to_datetime('2023-01-20')
    
    def test_get_crypto_subset(self, preprocessor):
        """Test cryptocurrency subset filtering"""
        # Create multi-crypto data
        data = pd.DataFrame({
            'crypto_name': ['Bitcoin', 'Ethereum', 'Litecoin'] * 10,
            'close': np.random.uniform(1000, 50000, 30),
            'date': pd.date_range('2023-01-01', periods=30, freq='D')
        })
        
        subset = preprocessor.get_crypto_subset(data, ['Bitcoin', 'Ethereum'])
        
        # Check that only selected cryptos are included
        assert set(subset['crypto_name'].unique()) == {'Bitcoin', 'Ethereum'}
        assert len(subset) == 20  # 2 cryptos * 10 records each
    
    def test_preprocess_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        # Add some data quality issues
        sample_data.loc[5, 'close'] = np.nan
        sample_data['Unnamed: 0'] = range(len(sample_data))
        
        processed_data, quality_report = preprocessor.preprocess_pipeline(sample_data)
        
        # Check that pipeline runs successfully
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(quality_report, dict)
        
        # Check that basic cleaning was applied
        assert 'Unnamed: 0' not in processed_data.columns
        
        # Check that time features were created
        assert 'year' in processed_data.columns
        
        # Check data quality
        assert processed_data['close'].isnull().sum() == 0

class TestDataValidation:
    """Test data validation and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        preprocessor = DataPreprocessor()
        empty_df = pd.DataFrame()
        
        # Should handle empty dataframe gracefully
        with pytest.raises((ValueError, KeyError)):
            preprocessor.basic_cleaning(empty_df)
    
    def test_single_row_dataframe(self):
        """Test handling of single row dataframe"""
        preprocessor = DataPreprocessor()
        single_row = pd.DataFrame({
            'date': ['2023-01-01'],
            'crypto_name': ['Bitcoin'],
            'open': [20000],
            'high': [21000],
            'low': [19000],
            'close': [20500],
            'volume': [1000000],
            'marketCap': [400000000000]
        })
        
        processed = preprocessor.basic_cleaning(single_row)
        assert len(processed) == 1
    
    def test_invalid_price_data(self):
        """Test handling of invalid price data"""
        preprocessor = DataPreprocessor()
        invalid_data = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02'],
            'crypto_name': ['Bitcoin', 'Bitcoin'],
            'open': [20000, -1000],  # Negative price
            'high': [21000, 25000],
            'low': [25000, 19000],   # Low > High
            'close': [20500, 24000],
            'volume': [1000000, 2000000],
            'marketCap': [400000000000, 450000000000]
        })
        
        quality_report = preprocessor.validate_data_quality(invalid_data)
        
        # Should detect price inconsistencies
        assert 'price_inconsistencies' in quality_report
        assert 'negative_values' in quality_report
