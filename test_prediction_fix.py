#!/usr/bin/env python3
"""
Quick test script to verify the prediction fix works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from app.utils.data_preprocessing import DataPreprocessor
from app.utils.feature_engineering import FeatureEngineer
from app.utils.model_utils import ModelManager

def test_prediction_fix():
    print("🧪 Testing prediction fix...")
    
    try:
        # Load a small sample of data
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data('dataset.csv')
        
        # Take just Bitcoin data for quick testing
        bitcoin_data = data[data['crypto_name'] == 'Bitcoin'].head(100).copy()
        print(f"Using {len(bitcoin_data)} rows of Bitcoin data for testing")
        
        # Create ModelManager instance
        model_manager = ModelManager()
        
        # Load the model
        print("Loading model...")
        model_manager.load_model()
        
        # Try to make a prediction
        print("Making prediction...")
        result = model_manager.predict(bitcoin_data, prediction_days=7)
        
        print("✅ Prediction successful!")
        print(f"   Volatility: {result['volatility']:.4f}")
        print(f"   Level: {result['level']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Recommendation: {result['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_fix()
    if success:
        print("\n🎉 Fix successful! The prediction error should now be resolved.")
    else:
        print("\n💥 Fix needs more work.")
