import pandas as pd
import numpy as np
import os
import pickle
import joblib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
from pathlib import Path

from app.models.ml_models import VolatilityPredictor
from app.utils.data_preprocessing import DataPreprocessor
from app.utils.feature_engineering import FeatureEngineer
from app.core.config import get_settings
from app.core.database import DatabaseManager

warnings.filterwarnings('ignore')

class ModelManager:
    """Centralized model management for training, prediction, and evaluation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.volatility_predictor = VolatilityPredictor()
        self.data_preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.db_manager = DatabaseManager()
        
        self.data = None
        self.processed_data = None
        self.model_loaded = False
        
        # Load data on initialization
        self.load_data()
    
    def load_data(self):
        """Load and prepare the dataset"""
        try:
            print("📊 Loading cryptocurrency dataset...")
            
            # Load raw data
            if os.path.exists(self.settings.DATASET_FILE):
                self.data = self.data_preprocessor.load_data(self.settings.DATASET_FILE)
                
                # Store raw data in database
                self.db_manager.insert_crypto_data(self.data)
                
                # Preprocess data
                self.processed_data, quality_report = self.data_preprocessor.preprocess_pipeline(
                    self.data
                )
                
                # Feature engineering
                self.processed_data = self.feature_engineer.feature_engineering_pipeline(
                    self.processed_data
                )
                
                print(f"✅ Data loaded and processed: {self.processed_data.shape}")
            else:
                print(f"❌ Dataset file not found: {self.settings.DATASET_FILE}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
    
    def train_model(self, retrain: bool = False) -> Dict[str, Any]:
        """Train the volatility prediction model"""
        if self.processed_data is None:
            raise ValueError("No data available for training")
        
        print("🚀 Starting model training...")
        
        training_start = datetime.now()
        
        # Prepare training data
        train_data = self.processed_data.dropna()
        
        if len(train_data) < self.settings.MIN_DATA_POINTS:
            raise ValueError(f"Insufficient data for training: {len(train_data)} < {self.settings.MIN_DATA_POINTS}")
        
        # Train model for each cryptocurrency separately
        training_results = {}
        
        for crypto_name in train_data['crypto_name'].unique():
            crypto_data = train_data[train_data['crypto_name'] == crypto_name].copy()
            
            if len(crypto_data) >= 100:  # Minimum data points per crypto
                print(f"🔄 Training model for {crypto_name}...")
                
                try:
                    results = self.volatility_predictor.train(crypto_data, crypto_name)
                    training_results[crypto_name] = results
                    
                    # Store training metrics in database
                    for model_name, metrics in results.items():
                        self.db_manager.insert_model_metrics(
                            f"{crypto_name}_{model_name}", 
                            metrics
                        )
                        
                except Exception as e:
                    print(f"❌ Error training model for {crypto_name}: {e}")
                    training_results[crypto_name] = {'error': str(e)}
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        # Save the trained model
        self.save_model()
        
        # Record training session
        training_record = {
            'model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_start': training_start.isoformat(),
            'training_end': training_end.isoformat(),
            'data_points': len(train_data),
            'accuracy': np.mean([
                np.mean([m.get('r2', 0) for m in crypto_results.values() if isinstance(crypto_results, dict)])
                for crypto_results in training_results.values()
                if isinstance(crypto_results, dict)
            ]),
            'r2_score': np.mean([
                np.mean([m.get('r2', 0) for m in crypto_results.values() if isinstance(crypto_results, dict)])
                for crypto_results in training_results.values()
                if isinstance(crypto_results, dict)
            ]),
            'mse': np.mean([
                np.mean([m.get('mse', 0) for m in crypto_results.values() if isinstance(crypto_results, dict)])
                for crypto_results in training_results.values()
                if isinstance(crypto_results, dict)
            ]),
            'config': {
                'cryptocurrencies': list(training_results.keys()),
                'training_duration': training_duration
            }
        }
        
        self.db_manager.insert_training_record(training_record)
        
        self.model_loaded = True
        
        print(f"✅ Model training completed in {training_duration:.2f} seconds")
        
        return {
            'status': 'success',
            'training_duration': training_duration,
            'cryptocurrencies_trained': len(training_results),
            'training_results': training_results,
            'model_version': training_record['model_version']
        }
    
    def predict(self, data: pd.DataFrame, prediction_days: int = 7) -> Dict[str, Any]:
        """Make volatility predictions"""
        if not self.model_loaded:
            try:
                self.load_model()
            except:
                raise ValueError("No trained model available. Please train the model first.")
        
        # Preprocess the input data
        processed_input = self.data_preprocessor.preprocess_pipeline(data)[0]
        processed_input = self.feature_engineer.feature_engineering_pipeline(processed_input)
        
        # Make prediction
        prediction = self.volatility_predictor.predict(processed_input, prediction_days)
        
        # Store prediction in database
        crypto_name = data['crypto_name'].iloc[0] if 'crypto_name' in data.columns else 'Unknown'
        prediction_data = {
            **prediction,
            'prediction_days': prediction_days
        }
        self.db_manager.insert_prediction(crypto_name, prediction_data)
        
        return prediction
    
    def get_latest_data(self, crypto_name: str, days: int = 100) -> Optional[pd.DataFrame]:
        """Get latest data for a specific cryptocurrency"""
        if self.processed_data is None:
            return None
        
        crypto_data = self.processed_data[
            self.processed_data['crypto_name'] == crypto_name
        ].copy()
        
        if len(crypto_data) == 0:
            return None
        
        # Sort by date and get latest data
        crypto_data = crypto_data.sort_values('date').tail(days)
        
        return crypto_data
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get market overview statistics"""
        if self.processed_data is None:
            return {'error': 'No data available'}
        
        df = self.processed_data
        
        # Calculate volatility for recent data
        recent_data = df.groupby('crypto_name').tail(30)  # Last 30 days per crypto
        
        volatility_stats = {}
        for crypto in recent_data['crypto_name'].unique():
            crypto_data = recent_data[recent_data['crypto_name'] == crypto]
            if 'returns' in crypto_data.columns and len(crypto_data) > 10:
                volatility = crypto_data['returns'].std() * np.sqrt(252)
                volatility_stats[crypto] = volatility
        
        if volatility_stats:
            avg_volatility = np.mean(list(volatility_stats.values()))
            high_volatility_count = sum(1 for v in volatility_stats.values() if v > self.settings.HIGH_VOLATILITY_THRESHOLD)
            low_volatility_count = sum(1 for v in volatility_stats.values() if v < self.settings.LOW_VOLATILITY_THRESHOLD)
            medium_volatility_count = len(volatility_stats) - high_volatility_count - low_volatility_count
        else:
            avg_volatility = 0
            high_volatility_count = 0
            low_volatility_count = 0
            medium_volatility_count = 0
        
        return {
            'total_cryptos': df['crypto_name'].nunique(),
            'avg_volatility': avg_volatility,
            'high_volatility_count': high_volatility_count,
            'medium_volatility_count': medium_volatility_count,
            'low_volatility_count': low_volatility_count,
            'data_points': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else None,
                'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else None
            },
            'last_updated': datetime.now().isoformat()
        }
    
    def get_crypto_metrics(self, crypto_name: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific cryptocurrency"""
        crypto_data = self.get_latest_data(crypto_name, days=365)  # Last year
        
        if crypto_data is None or len(crypto_data) == 0:
            return {'error': f'No data available for {crypto_name}'}
        
        # Calculate various metrics
        if 'returns' in crypto_data.columns:
            returns = crypto_data['returns'].dropna()
            
            metrics = {
                'crypto_name': crypto_name,
                'data_points': len(crypto_data),
                'date_range': {
                    'start': crypto_data['date'].min().strftime('%Y-%m-%d'),
                    'end': crypto_data['date'].max().strftime('%Y-%m-%d')
                },
                'price_metrics': {
                    'current_price': crypto_data['close'].iloc[-1],
                    'price_change_30d': ((crypto_data['close'].iloc[-1] / crypto_data['close'].iloc[-30]) - 1) * 100 if len(crypto_data) >= 30 else None,
                    'max_price': crypto_data['close'].max(),
                    'min_price': crypto_data['close'].min()
                },
                'volatility_metrics': {
                    'daily_volatility': returns.std(),
                    'annualized_volatility': returns.std() * np.sqrt(252),
                    'volatility_30d': returns.tail(30).std() * np.sqrt(252) if len(returns) >= 30 else None,
                    'max_drawdown': self._calculate_max_drawdown(crypto_data['close'])
                },
                'risk_metrics': {
                    'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                    'skewness': returns.skew(),
                    'kurtosis': returns.kurtosis(),
                    'var_95': returns.quantile(0.05),
                    'cvar_95': returns[returns <= returns.quantile(0.05)].mean()
                }
            }
        else:
            metrics = {
                'crypto_name': crypto_name,
                'error': 'Insufficient data for detailed metrics'
            }
        
        return metrics
    
    def retrain_model(self):
        """Retrain the model with latest data"""
        print("🔄 Starting model retraining...")
        
        # Reload data
        self.load_data()
        
        # Retrain model
        result = self.train_model(retrain=True)
        
        print("✅ Model retraining completed")
        return result
    
    def save_model(self):
        """Save the trained model"""
        model_path = self.settings.MODEL_FILE
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        self.volatility_predictor.save_model(model_path)
        
        # Save additional data
        additional_data = {
            'data_preprocessor': self.data_preprocessor,
            'feature_engineer': self.feature_engineer,
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'training_date': datetime.now().isoformat()
        }
        
        additional_path = model_path.replace('.joblib', '_additional.pkl')
        with open(additional_path, 'wb') as f:
            pickle.dump(additional_data, f)
        
        print(f"✅ Model saved to {model_path}")
    
    def load_model(self):
        """Load a previously trained model"""
        model_path = self.settings.MODEL_FILE
        
        if os.path.exists(model_path):
            self.volatility_predictor.load_model(model_path)
            
            # Load additional data
            additional_path = model_path.replace('.joblib', '_additional.pkl')
            if os.path.exists(additional_path):
                with open(additional_path, 'rb') as f:
                    additional_data = pickle.load(f)
                    self.data_preprocessor = additional_data.get('data_preprocessor', self.data_preprocessor)
                    self.feature_engineer = additional_data.get('feature_engineer', self.feature_engineer)
            
            self.model_loaded = True
            print("✅ Model loaded successfully")
        else:
            print(f"❌ Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded and self.volatility_predictor.is_trained
    
    def get_feature_importance(self, crypto_name: str = None, model_name: str = 'random_forest') -> Dict[str, float]:
        """Get feature importance from the trained model"""
        if not self.is_model_loaded():
            return {}
        
        return self.volatility_predictor.get_feature_importance(model_name)
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.backends.backend_pdf import PdfPages
            
            report_path = os.path.join(self.settings.DATA_PATH, "volatility_analysis_report.pdf")
            
            with PdfPages(report_path) as pdf:
                # Market overview
                market_stats = self.get_market_overview()
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle('Cryptocurrency Market Overview', fontsize=16)
                
                # Volatility distribution
                if self.processed_data is not None:
                    recent_data = self.processed_data.groupby('crypto_name').tail(30)
                    volatilities = []
                    crypto_names = []
                    
                    for crypto in recent_data['crypto_name'].unique():
                        crypto_data = recent_data[recent_data['crypto_name'] == crypto]
                        if 'returns' in crypto_data.columns and len(crypto_data) > 10:
                            vol = crypto_data['returns'].std() * np.sqrt(252)
                            volatilities.append(vol)
                            crypto_names.append(crypto)
                    
                    if volatilities:
                        axes[0, 0].hist(volatilities, bins=20, alpha=0.7)
                        axes[0, 0].set_title('Volatility Distribution')
                        axes[0, 0].set_xlabel('Annualized Volatility')
                        axes[0, 0].set_ylabel('Frequency')
                        
                        # Top 10 most volatile
                        top_volatile = sorted(zip(crypto_names, volatilities), key=lambda x: x[1], reverse=True)[:10]
                        names, vols = zip(*top_volatile)
                        
                        axes[0, 1].barh(range(len(names)), vols)
                        axes[0, 1].set_yticks(range(len(names)))
                        axes[0, 1].set_yticklabels(names)
                        axes[0, 1].set_title('Top 10 Most Volatile')
                        axes[0, 1].set_xlabel('Volatility')
                
                # Market statistics
                stats_text = f"""
                Total Cryptocurrencies: {market_stats.get('total_cryptos', 'N/A')}
                Average Volatility: {market_stats.get('avg_volatility', 0):.3f}
                High Volatility Assets: {market_stats.get('high_volatility_count', 0)}
                Medium Volatility Assets: {market_stats.get('medium_volatility_count', 0)}
                Low Volatility Assets: {market_stats.get('low_volatility_count', 0)}
                Total Data Points: {market_stats.get('data_points', 0):,}
                """
                
                axes[1, 0].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
                axes[1, 0].set_xlim(0, 1)
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].axis('off')
                axes[1, 0].set_title('Market Statistics')
                
                # Model performance
                if self.is_model_loaded():
                    feature_importance = self.get_feature_importance()
                    if feature_importance:
                        top_features = dict(list(feature_importance.items())[:10])
                        
                        axes[1, 1].barh(range(len(top_features)), list(top_features.values()))
                        axes[1, 1].set_yticks(range(len(top_features)))
                        axes[1, 1].set_yticklabels(list(top_features.keys()), fontsize=8)
                        axes[1, 1].set_title('Top 10 Feature Importance')
                        axes[1, 1].set_xlabel('Importance')
                
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
            
            print(f"✅ Report generated: {report_path}")
            return report_path
            
        except ImportError:
            # Fallback to text report if matplotlib not available
            report_path = os.path.join(self.settings.DATA_PATH, "volatility_analysis_report.txt")
            
            with open(report_path, 'w') as f:
                f.write("Cryptocurrency Volatility Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                market_stats = self.get_market_overview()
                f.write("Market Overview:\n")
                for key, value in market_stats.items():
                    f.write(f"  {key}: {value}\n")
                
                f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            return report_path
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility != 0 else 0
