import os
from functools import lru_cache
from typing import Optional

class Settings:
    """Application settings and configuration"""
    
    # API Configuration
    API_TITLE: str = "Cryptocurrency Volatility Prediction API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "ML-powered API for predicting cryptocurrency market volatility"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Database Configuration
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", "sqlite:///data/crypto_volatility.db")
    
    # Data Configuration
    DATA_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    RAW_DATA_PATH: str = os.path.join(DATA_PATH, "raw")
    PROCESSED_DATA_PATH: str = os.path.join(DATA_PATH, "processed")
    MODELS_PATH: str = os.path.join(DATA_PATH, "models")
    
    # Dataset Configuration
    DATASET_FILE: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset.csv")
    
    # Model Configuration
    MODEL_FILE: str = os.path.join(MODELS_PATH, "volatility_model.joblib")
    SCALER_FILE: str = os.path.join(MODELS_PATH, "scalers.joblib")
    
    # Training Configuration
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_SPLIT: float = 0.2
    RANDOM_STATE: int = 42
    
    # Feature Engineering
    VOLATILITY_WINDOW: int = 20
    MA_WINDOWS: list = [5, 10, 20, 50]
    RSI_WINDOW: int = 14
    BOLLINGER_WINDOW: int = 20
    BOLLINGER_STD: float = 2.0
    
    # Prediction Configuration
    MAX_PREDICTION_DAYS: int = 30
    MIN_PREDICTION_DAYS: int = 1
    DEFAULT_PREDICTION_DAYS: int = 7
    
    # Volatility Thresholds
    LOW_VOLATILITY_THRESHOLD: float = 0.02
    HIGH_VOLATILITY_THRESHOLD: float = 0.05
    
    # API Rate Limiting
    RATE_LIMIT_CALLS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # 1 hour
    
    # Caching
    CACHE_TTL: int = 300  # 5 minutes
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "app.log")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALLOWED_HOSTS: list = ["*"]
    
    # External APIs (if needed)
    COINAPI_KEY: Optional[str] = os.getenv("COINAPI_KEY")
    ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_KEY")
    
    # Model Performance Thresholds
    MIN_MODEL_ACCURACY: float = 0.6
    MIN_MODEL_R2: float = 0.4
    
    # Data Quality
    MIN_DATA_POINTS: int = 100
    MAX_MISSING_VALUES_RATIO: float = 0.1
    
    def __init__(self):
        # Create necessary directories
        os.makedirs(self.DATA_PATH, exist_ok=True)
        os.makedirs(self.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
