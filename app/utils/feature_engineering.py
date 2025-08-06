import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import ta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Advanced feature engineering for cryptocurrency volatility prediction"""
    
    def __init__(self):
        self.feature_groups = {
            'price_features': [],
            'volume_features': [],
            'technical_indicators': [],
            'volatility_features': [],
            'statistical_features': [],
            'market_features': []
        }
    
    def create_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        df = data.copy()
        
        # Basic price features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['body_size'] = np.abs(df['open'] - df['close']) / df['close']
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['overnight_returns'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['intraday_returns'] = (df['close'] - df['open']) / df['open']
        
        # Cumulative returns
        for period in [5, 10, 20]:
            df[f'cumulative_returns_{period}d'] = df['returns'].rolling(window=period).apply(
                lambda x: (1 + x).prod() - 1
            )
        
        self.feature_groups['price_features'] = [
            'price_range', 'price_gap', 'upper_shadow', 'lower_shadow', 'body_size',
            'high_low_ratio', 'open_close_ratio', 'returns', 'log_returns',
            'overnight_returns', 'intraday_returns'
        ] + [f'cumulative_returns_{period}d' for period in [5, 10, 20]]
        
        return df
    
    def create_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create moving average features"""
        df = data.copy()
        
        # Simple moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'sma_ratio_{window}'] = df['close'] / df[f'sma_{window}']
            df[f'sma_distance_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
        
        # Exponential moving averages
        for window in [12, 26, 50]:
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'ema_ratio_{window}'] = df['close'] / df[f'ema_{window}']
        
        # Moving average convergence divergence (MACD)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Moving average slopes
        for window in [10, 20, 50]:
            df[f'sma_slope_{window}'] = df[f'sma_{window}'].diff(5) / df[f'sma_{window}'].shift(5)
        
        return df
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis indicators"""
        df = data.copy()
        
        # RSI (Relative Strength Index)
        for window in [14, 21, 30]:
            df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
        
        # Bollinger Bands
        for window in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], window)
            df[f'bb_upper_{window}'] = bb_upper
            df[f'bb_middle_{window}'] = bb_middle
            df[f'bb_lower_{window}'] = bb_lower
            df[f'bb_width_{window}'] = (bb_upper - bb_lower) / bb_middle
            df[f'bb_position_{window}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df['high'], df['low'], df['close'])
        
        # Average True Range (ATR)
        df['atr'] = self._calculate_atr(df['high'], df['low'], df['close'])
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Commodity Channel Index (CCI)
        df['cci'] = self._calculate_cci(df['high'], df['low'], df['close'])
        
        # Money Flow Index (MFI)
        if 'volume' in df.columns:
            df['mfi'] = self._calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
        
        self.feature_groups['technical_indicators'] = [
            col for col in df.columns if any(indicator in col for indicator in 
            ['rsi', 'bb_', 'stoch', 'williams', 'atr', 'cci', 'mfi'])
        ]
        
        return df
    
    def create_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        if 'volume' not in data.columns:
            return data
        
        df = data.copy()
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # Volume price trend
        df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        # On-Balance Volume (OBV)
        df['obv'] = self._calculate_obv(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10)
        
        # Price Volume Trend
        df['pvt'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1) * df['volume']).cumsum()
        
        # Volume Weighted Average Price (VWAP)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['vwap_ratio'] = df['close'] / df['vwap']
        
        self.feature_groups['volume_features'] = [
            col for col in df.columns if 'volume' in col or col in ['vpt', 'obv', 'pvt', 'vwap', 'vwap_ratio']
        ]
        
        return df
    
    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        df = data.copy()
        
        # Historical volatility (different windows)
        for window in [5, 10, 20, 30, 60]:
            df[f'volatility_{window}d'] = df['returns'].rolling(window=window).std() * np.sqrt(252)
            df[f'volatility_ratio_{window}d'] = df[f'volatility_{window}d'] / df[f'volatility_{window}d'].rolling(window=60).mean()
        
        # Garman-Klass volatility (uses OHLC)
        df['gk_volatility'] = np.sqrt(
            252 * (
                0.5 * (np.log(df['high'] / df['low'])) ** 2 -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open'])) ** 2
            )
        )
        
        # Rogers-Satchell volatility
        df['rs_volatility'] = np.sqrt(
            252 * (
                np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) +
                np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
            ).rolling(window=20).mean()
        )
        
        # Parkinson volatility
        df['parkinson_volatility'] = np.sqrt(
            252 * (np.log(df['high'] / df['low'])) ** 2 / (4 * np.log(2))
        ).rolling(window=20).mean()
        
        # Volatility clustering (GARCH-like)
        df['volatility_persistence'] = df['volatility_20d'].rolling(window=10).corr(df['volatility_20d'].shift(1))
        
        # Volatility skewness and kurtosis
        for window in [20, 60]:
            df[f'returns_skewness_{window}d'] = df['returns'].rolling(window=window).skew()
            df[f'returns_kurtosis_{window}d'] = df['returns'].rolling(window=window).kurt()
        
        self.feature_groups['volatility_features'] = [
            col for col in df.columns if 'volatility' in col or 'skewness' in col or 'kurtosis' in col
        ]
        
        return df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        df = data.copy()
        
        # Rolling statistics for returns
        for window in [5, 10, 20, 60]:
            df[f'returns_mean_{window}d'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}d'] = df['returns'].rolling(window=window).std()
            df[f'returns_min_{window}d'] = df['returns'].rolling(window=window).min()
            df[f'returns_max_{window}d'] = df['returns'].rolling(window=window).max()
            df[f'returns_median_{window}d'] = df['returns'].rolling(window=window).median()
            df[f'returns_q25_{window}d'] = df['returns'].rolling(window=window).quantile(0.25)
            df[f'returns_q75_{window}d'] = df['returns'].rolling(window=window).quantile(0.75)
        
        # Momentum indicators
        for period in [5, 10, 20]:
            df[f'momentum_{period}d'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}d'] = df['close'].pct_change(periods=period)
        
        # Autocorrelation
        for lag in [1, 5, 10]:
            df[f'returns_autocorr_lag{lag}'] = df['returns'].rolling(window=60).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        
        # Z-score normalization
        for window in [20, 60]:
            rolling_mean = df['close'].rolling(window=window).mean()
            rolling_std = df['close'].rolling(window=window).std()
            df[f'price_zscore_{window}d'] = (df['close'] - rolling_mean) / rolling_std
        
        self.feature_groups['statistical_features'] = [
            col for col in df.columns if any(stat in col for stat in 
            ['mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'momentum', 'roc', 'autocorr', 'zscore'])
        ]
        
        return df
    
    def create_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create market-wide features"""
        df = data.copy()
        
        if 'marketCap' in df.columns:
            # Market cap features
            df['market_cap_change'] = df['marketCap'].pct_change()
            df['market_cap_ma_20'] = df['marketCap'].rolling(window=20).mean()
            df['market_cap_ratio'] = df['marketCap'] / df['market_cap_ma_20']
            
            # Market dominance (if multiple cryptos)
            if df['crypto_name'].nunique() > 1:
                total_market_cap = df.groupby('date')['marketCap'].sum()
                df['market_dominance'] = df['marketCap'] / df['date'].map(total_market_cap)
        
        # Price efficiency (random walk test)
        df['price_efficiency'] = df['returns'].rolling(window=20).apply(
            lambda x: self._calculate_hurst_exponent(x.values) if len(x) == 20 else np.nan
        )
        
        self.feature_groups['market_features'] = [
            col for col in df.columns if 'market' in col or 'dominance' in col or 'efficiency' in col
        ]
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, 
                          target_col: str = 'returns',
                          lags: List[int] = [1, 2, 3, 5, 7, 14]) -> pd.DataFrame:
        """Create lagged features"""
        df = data.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different indicators"""
        df = data.copy()
        
        # Price and volume interactions
        if all(col in df.columns for col in ['returns', 'volume_ratio_20']):
            df['price_volume_interaction'] = df['returns'] * df['volume_ratio_20']
        
        # Technical indicator interactions
        if all(col in df.columns for col in ['rsi_14', 'bb_position_20']):
            df['rsi_bb_interaction'] = df['rsi_14'] * df['bb_position_20']
        
        # Volatility and momentum interaction
        if all(col in df.columns for col in ['volatility_20d', 'momentum_10d']):
            df['volatility_momentum_interaction'] = df['volatility_20d'] * df['momentum_10d']
        
        return df
    
    def feature_engineering_pipeline(self, data: pd.DataFrame, 
                                   config: Optional[Dict] = None) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        if config is None:
            # 🚀 OPTIMIZED CONFIG: Reduce features for faster processing
            config = {
                'price_features': True,
                'moving_averages': True,
                'technical_indicators': True,
                'volume_features': False,  # Skip volume features for speed
                'volatility_features': True,
                'statistical_features': False,  # Skip statistical features for speed
                'market_features': False,  # Skip market features for speed
                'lag_features': False,  # Skip lag features for speed
                'interaction_features': False  # Skip interaction features for speed
            }
        
        df = data.copy()
        
        print("🔄 Starting optimized feature engineering pipeline...")
        
        # Group by crypto to ensure features are calculated per cryptocurrency
        crypto_dfs = []
        
        for crypto_name in df['crypto_name'].unique():
            crypto_df = df[df['crypto_name'] == crypto_name].copy().sort_values('date')
            
            print(f"   ⚡ Processing {crypto_name}...")
            
            if config.get('price_features', True):
                crypto_df = self.create_price_features(crypto_df)
            
            if config.get('moving_averages', True):
                crypto_df = self.create_moving_averages(crypto_df)
            
            if config.get('technical_indicators', True):
                crypto_df = self.create_technical_indicators(crypto_df)
            
            if config.get('volume_features', False):
                crypto_df = self.create_volume_features(crypto_df)
            
            if config.get('volatility_features', True):
                crypto_df = self.create_volatility_features(crypto_df)
            
            if config.get('statistical_features', False):
                crypto_df = self.create_statistical_features(crypto_df)
            
            if config.get('market_features', False):
                crypto_df = self.create_market_features(crypto_df)
            
            if config.get('lag_features', False):
                crypto_df = self.create_lag_features(crypto_df)
            
            crypto_dfs.append(crypto_df)
        
        
        # Combine all crypto dataframes
        df = pd.concat(crypto_dfs, ignore_index=True)
        
        # Create interaction features after combining
        if config.get('interaction_features', True):
            df = self.create_interaction_features(df)
        
        # Remove features with too many NaN values
        threshold = 0.7  # Remove features with more than 70% NaN values
        df = df.loc[:, df.isnull().mean() < threshold]
        
        print(f"✅ Feature engineering completed: {df.shape[1]} features created")
        
        return df
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, ma, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Calculate Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mad = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    def _calculate_mfi(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14):
        """Calculate Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    def _calculate_obv(self, close: pd.Series, volume: pd.Series):
        """Calculate On-Balance Volume"""
        obv = volume.copy()
        obv[close < close.shift(1)] *= -1
        return obv.cumsum()
    
    def _calculate_hurst_exponent(self, time_series):
        """Calculate Hurst exponent for price efficiency"""
        try:
            lags = range(2, min(20, len(time_series)//2))
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return np.nan
