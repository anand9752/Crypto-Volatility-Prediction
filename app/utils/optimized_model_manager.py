"""
Optimized Model Manager that loads pre-processed data and model only once
"""

import pandas as pd
import joblib
import os
from typing import Dict, Any, Optional
from datetime import datetime
import pickle

from models.ml_models import VolatilityPredictor
from core.config import Settings

class OptimizedModelManager:
    """Optimized model manager that loads everything once at startup"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.processed_data: Optional[pd.DataFrame] = None
        self.predictor: Optional[VolatilityPredictor] = None
        self.metadata: Optional[Dict] = None
        self.model_loaded = False
        
        # Load everything once at initialization
        self._load_processed_data()
        self._load_model()
        self._load_metadata()
    
    def _load_processed_data(self):
        """Load pre-processed data from file"""
        try:
            processed_data_path = "data/processed/processed_crypto_data.joblib"
            
            if os.path.exists(processed_data_path):
                self.processed_data = joblib.load(processed_data_path)
                print(f"✅ Processed data loaded: {self.processed_data.shape[0]} rows, {self.processed_data.shape[1]} features")
            else:
                print(f"❌ Processed data not found at: {processed_data_path}")
                print("💡 Run 'python prepare_data_once.py' first to process the data")
                
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
    
    def _load_model(self):
        """Load pre-trained model from file"""
        try:
            model_path = self.settings.MODEL_FILE
            
            if os.path.exists(model_path):
                self.predictor = VolatilityPredictor()
                self.predictor.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Model loaded from: {model_path}")
            else:
                print(f"❌ Model not found at: {model_path}")
                print("💡 Run 'python prepare_data_once.py' first to train the model")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def _load_metadata(self):
        """Load setup metadata"""
        try:
            metadata_path = "data/processed/setup_metadata.joblib"
            
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                print(f"✅ Metadata loaded - Setup date: {self.metadata.get('setup_date', 'Unknown')}")
            else:
                print("⚠️ No metadata found")
                
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """Get the loaded processed data"""
        return self.processed_data
    
    def is_ready(self) -> bool:
        """Check if both data and model are ready"""
        return self.processed_data is not None and self.model_loaded
    
    def predict(self, data: pd.DataFrame, prediction_days: int = 7) -> Dict[str, Any]:
        """Make volatility predictions using the loaded model"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Run prepare_data_once.py first.")
        
        return self.predictor.predict(data, prediction_days)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model and data"""
        if not self.is_ready():
            return {"error": "Model or data not ready"}
        
        return {
            "data_shape": self.processed_data.shape,
            "cryptocurrencies": list(self.processed_data['crypto_name'].unique()) if 'crypto_name' in self.processed_data.columns else [],
            "model_loaded": self.model_loaded,
            "setup_metadata": self.metadata,
            "ready": self.is_ready()
        }
    
    def get_latest_data(self, crypto_name: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get latest data for a specific cryptocurrency"""
        if self.processed_data is None:
            return None
        
        # Handle different column name possibilities
        name_col = 'crypto_name' if 'crypto_name' in self.processed_data.columns else 'Name'
        
        if name_col not in self.processed_data.columns:
            return None
        
        crypto_data = self.processed_data[self.processed_data[name_col] == crypto_name]
        return crypto_data.tail(days) if not crypto_data.empty else None
