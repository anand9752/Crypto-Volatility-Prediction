import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import pickle
import warnings
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class VolatilityPredictor:
    """
    Advanced machine learning model for cryptocurrency volatility prediction
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.model_metrics = {}
        self.is_trained = False
        
        # Model configurations - OPTIMIZED for faster training
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=50,  # Reduced from 100
                    max_depth=8,      # Reduced from 10
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1  # Changed from -1 to 1 to avoid joblib issues on Windows
                ),
                'params': {
                    'n_estimators': [30, 50],        # Reduced grid search
                    'max_depth': [6, 8],             # Reduced grid search
                    'min_samples_split': [3, 5]      # Reduced grid search
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=50,    # Reduced from 100
                    learning_rate=0.1,
                    max_depth=4,        # Reduced from 6
                    random_state=42
                ),
                'params': {
                    'n_estimators': [30, 50],        # Reduced grid search
                    'learning_rate': [0.1],          # Fixed value for speed
                    'max_depth': [4, 6]              # Reduced grid search
                }
            }
        }
    
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate rolling volatility using standard deviation of returns"""
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for volatility prediction"""
        features_df = data.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['price_gap'] = (features_df['open'] - features_df['close'].shift(1)) / features_df['close'].shift(1)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'ma_{window}'] = features_df['close'].rolling(window=window).mean()
            features_df[f'ma_ratio_{window}'] = features_df['close'] / features_df[f'ma_{window}']
        
        # Technical indicators
        features_df['rsi'] = self._calculate_rsi(features_df['close'])
        features_df['bollinger_upper'], features_df['bollinger_lower'] = self._calculate_bollinger_bands(features_df['close'])
        features_df['bollinger_width'] = (features_df['bollinger_upper'] - features_df['bollinger_lower']) / features_df['close']
        
        # Volume features
        features_df['volume_ma'] = features_df['volume'].rolling(window=20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma']
        features_df['price_volume'] = features_df['close'] * features_df['volume']
        
        # Market cap features
        features_df['market_cap_change'] = features_df['marketCap'].pct_change()
        features_df['market_cap_ma'] = features_df['marketCap'].rolling(window=20).mean()
        
        # Volatility features (lagged)
        for lag in [1, 2, 3, 5, 7]:
            features_df[f'volatility_lag_{lag}'] = features_df['returns'].rolling(window=20).std().shift(lag)
        
        # Time-based features
        features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
        features_df['month'] = pd.to_datetime(features_df['date']).dt.month
        features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter
        
        # Target variable - future volatility
        features_df['target_volatility'] = features_df['returns'].rolling(window=20).std().shift(-7)
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
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
        return upper_band, lower_band
    
    def train(self, data: pd.DataFrame, crypto_name: str = None) -> Dict[str, Any]:
        """Train volatility prediction models"""
        print(f"Training volatility prediction model for {crypto_name or 'all cryptocurrencies'}...")
        
        # Prepare features
        features_df = self.prepare_features(data)
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        if len(features_df) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")
        
        # Select feature columns
        feature_columns = [col for col in features_df.columns if col not in 
                          ['date', 'crypto_name', 'timestamp', 'target_volatility', 'open', 'high', 'low', 'close']]
        
        X = features_df[feature_columns]
        y = features_df['target_volatility']
        
        self.feature_names = feature_columns
        
        # Split data chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[crypto_name or 'default'] = scaler
        
        # Train models
        results = {}
        for model_name, config in self.model_configs.items():
            print(f"Training {model_name}...")
            
            # Grid search with time series cross-validation - OPTIMIZED
            tscv = TimeSeriesSplit(n_splits=2)  # Reduced from 3 to 2 for speed
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=1  # Changed from -1 to 1 to avoid joblib issues on Windows
            )
            
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = best_model.predict(X_test_scaled)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'best_params': grid_search.best_params_
            }
            
            self.models[model_name] = best_model
            self.model_metrics[model_name] = metrics
            results[model_name] = metrics
            
            print(f"{model_name} - MSE: {metrics['mse']:.6f}, R2: {metrics['r2']:.3f}")
        
        self.is_trained = True
        return results
    
    def predict(self, data: pd.DataFrame, days_ahead: int = 7) -> Dict[str, Any]:
        """Predict volatility for the specified number of days ahead"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Check if data already has engineered features (from FeatureEngineer)
        feature_cols = [col for col in data.columns if col not in 
                       ['date', 'crypto_name', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'marketCap']]
        
        if len(feature_cols) > 50:  # Data already has engineered features
            print(f"Using pre-engineered features: {len(feature_cols)} features available")
            
            # Use only the features that exist in both the data and the model's expected features
            available_features = [f for f in self.feature_names if f in data.columns]
            
            if len(available_features) < 10:  # Need at least some features
                print(f"Warning: Only {len(available_features)} features match. Using available features.")
                # Use the first N features that are available
                available_features = feature_cols[:min(len(self.feature_names), len(feature_cols))]
            
            # Get latest features
            latest_features = data[available_features].iloc[-1:].values
            
            # Pad or truncate to match expected feature count
            if latest_features.shape[1] < len(self.feature_names):
                # Pad with zeros
                padding = np.zeros((1, len(self.feature_names) - latest_features.shape[1]))
                latest_features = np.hstack([latest_features, padding])
            elif latest_features.shape[1] > len(self.feature_names):
                # Truncate
                latest_features = latest_features[:, :len(self.feature_names)]
                
        else:
            # Fall back to creating features using the old method
            print("Creating features using VolatilityPredictor's prepare_features method")
            features_df = self.prepare_features(data)
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                raise ValueError("No valid data for prediction")
            
            latest_features = features_df[self.feature_names].iloc[-1:].values
        
        # Scale features
        scaler = list(self.scalers.values())[0]  # Use default scaler
        latest_features_scaled = scaler.transform(latest_features)
        
        # Make predictions with ensemble
        predictions = {}
        for model_name, model in self.models.items():
            pred = model.predict(latest_features_scaled)[0]
            predictions[model_name] = pred
        
        # Ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Calculate confidence based on prediction variance
        prediction_std = np.std(list(predictions.values()))
        confidence = max(0.5, 1 - (prediction_std / ensemble_pred) if ensemble_pred > 0 else 0.5)
        
        # Classify volatility level
        volatility_level = self._classify_volatility(ensemble_pred)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(ensemble_pred, volatility_level, confidence)
        
        return {
            'volatility': ensemble_pred,
            'level': volatility_level,
            'confidence': confidence,
            'recommendation': recommendation,
            'individual_predictions': predictions
        }
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility into Low, Medium, High categories"""
        if volatility < 0.02:
            return "Low"
        elif volatility < 0.05:
            return "Medium"
        else:
            return "High"
    
    def _generate_recommendation(self, volatility: float, level: str, confidence: float) -> str:
        """Generate trading recommendation based on volatility prediction"""
        if level == "Low" and confidence > 0.7:
            return "Low risk - suitable for conservative strategies and large positions"
        elif level == "Medium":
            return "Moderate risk - consider position sizing and stop-loss strategies"
        elif level == "High" and confidence > 0.7:
            return "High risk - recommended for experienced traders only, use small positions"
        else:
            return "Uncertain market conditions - exercise extreme caution and consider waiting"
    
    def save_model(self, filepath: str):
        """Save trained models and scalers"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'model_metrics': self.model_metrics,
            'is_trained': self.is_trained,
            'timestamp': datetime.now()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models and scalers"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.model_metrics = model_data['model_metrics']
            self.is_trained = model_data['is_trained']
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance for interpretability"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
