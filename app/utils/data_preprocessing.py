import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import os

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing utilities for cryptocurrency data"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.data_stats = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load cryptocurrency data from CSV file"""
        try:
            data = pd.read_csv(filepath)
            print(f"✅ Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")
            return data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform basic data cleaning operations"""
        df = data.copy()
        
        # Remove unnamed index column if present
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Remove duplicates
        before_count = len(df)
        df = df.drop_duplicates(subset=['crypto_name', 'date'], keep='last')
        after_count = len(df)
        
        if before_count != after_count:
            print(f"🔄 Removed {before_count - after_count} duplicate records")
        
        # Sort by crypto name and date
        df = df.sort_values(['crypto_name', 'date']).reset_index(drop=True)
        
        print(f"✅ Basic cleaning completed: {df.shape[0]} rows remaining")
        return df
    
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = data.copy()
        
        # Check for missing values
        missing_summary = df.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            print(f"📊 Missing values found in columns: {missing_cols.to_dict()}")
        
        # Handle missing values for numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
        
        for col in numeric_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                if strategy == 'interpolate':
                    # Group by crypto and interpolate
                    df[col] = df.groupby('crypto_name')[col].apply(
                        lambda x: x.interpolate(method='linear', limit_direction='both')
                    )
                elif strategy == 'forward_fill':
                    df[col] = df.groupby('crypto_name')[col].fillna(method='ffill')
                elif strategy == 'mean':
                    df[col] = df.groupby('crypto_name')[col].fillna(
                        df.groupby('crypto_name')[col].transform('mean')
                    )
        
        # Final check and remove rows with still missing critical values
        critical_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=critical_cols)
        
        print(f"✅ Missing value handling completed: {df.shape[0]} rows remaining")
        return df
    
    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect and optionally remove outliers"""
        df = data.copy()
        outlier_counts = {}
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
        
        for crypto in df['crypto_name'].unique():
            crypto_data = df[df['crypto_name'] == crypto].copy()
            outlier_mask = pd.Series([False] * len(crypto_data), index=crypto_data.index)
            
            for col in numeric_columns:
                if col in crypto_data.columns:
                    if method == 'iqr':
                        Q1 = crypto_data[col].quantile(0.25)
                        Q3 = crypto_data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        col_outliers = (crypto_data[col] < lower_bound) | (crypto_data[col] > upper_bound)
                        outlier_mask |= col_outliers
                    
                    elif method == 'zscore':
                        z_scores = np.abs((crypto_data[col] - crypto_data[col].mean()) / crypto_data[col].std())
                        col_outliers = z_scores > 3
                        outlier_mask |= col_outliers
            
            outlier_counts[crypto] = outlier_mask.sum()
        
        total_outliers = sum(outlier_counts.values())
        print(f"📊 Detected {total_outliers} outliers across all cryptocurrencies")
        
        # Store outlier information
        df['is_outlier'] = False
        for crypto, count in outlier_counts.items():
            if count > 0:
                crypto_data = df[df['crypto_name'] == crypto].copy()
                # Recompute outliers for marking
                outlier_mask = pd.Series([False] * len(crypto_data), index=crypto_data.index)
                
                for col in numeric_columns:
                    if col in crypto_data.columns:
                        if method == 'iqr':
                            Q1 = crypto_data[col].quantile(0.25)
                            Q3 = crypto_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            col_outliers = (crypto_data[col] < lower_bound) | (crypto_data[col] > upper_bound)
                            outlier_mask |= col_outliers
                
                df.loc[outlier_mask.index[outlier_mask], 'is_outlier'] = True
        
        return df
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """Validate data quality and generate quality report"""
        df = data.copy()
        
        quality_report = {
            'total_records': len(df),
            'total_cryptocurrencies': df['crypto_name'].nunique(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else None,
                'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else None,
                'days': (df['date'].max() - df['date'].min()).days if 'date' in df.columns else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'outliers': df['is_outlier'].sum() if 'is_outlier' in df.columns else 0,
        }
        
        # Check for price consistency (high >= low, etc.)
        price_inconsistencies = 0
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            price_inconsistencies = (
                (df['high'] < df['low']).sum() +
                (df['high'] < df['open']).sum() +
                (df['high'] < df['close']).sum() +
                (df['low'] > df['open']).sum() +
                (df['low'] > df['close']).sum()
            )
        
        quality_report['price_inconsistencies'] = price_inconsistencies
        
        # Check for negative values in price/volume columns
        negative_values = {}
        for col in ['open', 'high', 'low', 'close', 'volume', 'marketCap']:
            if col in df.columns:
                negative_values[col] = (df[col] < 0).sum()
        
        quality_report['negative_values'] = negative_values
        
        # Data completeness by cryptocurrency
        crypto_completeness = {}
        for crypto in df['crypto_name'].unique():
            crypto_data = df[df['crypto_name'] == crypto]
            completeness = 1 - (crypto_data.isnull().sum().sum() / (len(crypto_data) * len(crypto_data.columns)))
            crypto_completeness[crypto] = round(completeness, 3)
        
        quality_report['crypto_completeness'] = crypto_completeness
        
        print("📊 Data Quality Report:")
        print(f"   Total Records: {quality_report['total_records']:,}")
        print(f"   Cryptocurrencies: {quality_report['total_cryptocurrencies']}")
        print(f"   Date Range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
        print(f"   Price Inconsistencies: {quality_report['price_inconsistencies']}")
        print(f"   Outliers: {quality_report['outliers']}")
        
        return quality_report
    
    def normalize_features(self, data: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
        """Normalize numerical features"""
        df = data.copy()
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'marketCap']
        columns_to_scale = [col for col in numeric_columns if col in df.columns]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Scale within each cryptocurrency group
        for crypto in df['crypto_name'].unique():
            mask = df['crypto_name'] == crypto
            crypto_data = df.loc[mask, columns_to_scale]
            
            if len(crypto_data) > 1:  # Only scale if we have multiple data points
                scaled_data = scaler.fit_transform(crypto_data)
                df.loc[mask, columns_to_scale] = scaled_data
                
                # Store scaler for this crypto
                self.scalers[crypto] = scaler
        
        print(f"✅ Feature normalization completed using {method} scaling")
        return df
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = data.copy()
        
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical encoding for periodic features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            print("✅ Time-based features created")
        
        return df
    
    def filter_by_date_range(self, data: pd.DataFrame, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Filter data by date range"""
        df = data.copy()
        
        if start_date:
            start_date = pd.to_datetime(start_date)
            df = df[df['date'] >= start_date]
        
        if end_date:
            end_date = pd.to_datetime(end_date)
            df = df[df['date'] <= end_date]
        
        print(f"✅ Date filtering completed: {len(df)} rows remaining")
        return df
    
    def get_crypto_subset(self, data: pd.DataFrame, 
                         crypto_names: List[str]) -> pd.DataFrame:
        """Get data for specific cryptocurrencies"""
        df = data.copy()
        df = df[df['crypto_name'].isin(crypto_names)]
        
        print(f"✅ Filtered to {len(crypto_names)} cryptocurrencies: {len(df)} rows")
        return df
    
    def preprocess_pipeline(self, data: pd.DataFrame, 
                          config: Optional[Dict] = None) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        if config is None:
            config = {
                'handle_missing': True,
                'missing_strategy': 'interpolate',
                'detect_outliers': True,
                'outlier_method': 'iqr',
                'normalize': False,
                'normalization_method': 'robust',
                'create_time_features': True
            }
        
        df = data.copy()
        
        print("🔄 Starting preprocessing pipeline...")
        
        # Basic cleaning
        df = self.basic_cleaning(df)
        
        # Handle missing values
        if config.get('handle_missing', True):
            df = self.handle_missing_values(df, config.get('missing_strategy', 'interpolate'))
        
        # Detect outliers
        if config.get('detect_outliers', True):
            df = self.detect_outliers(df, config.get('outlier_method', 'iqr'))
        
        # Create time features
        if config.get('create_time_features', True):
            df = self.create_time_features(df)
        
        # Normalize features
        if config.get('normalize', False):
            df = self.normalize_features(df, config.get('normalization_method', 'robust'))
        
        # Generate quality report
        quality_report = self.validate_data_quality(df)
        
        print("✅ Preprocessing pipeline completed successfully!")
        
        return df, quality_report
