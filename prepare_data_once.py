#!/usr/bin/env python3
"""
One-time setup script to process data and train model
Run this once to prepare everything for the application
"""

import os
import sys
import pandas as pd
import joblib
from datetime import datetime

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.data_preprocessing import DataPreprocessor
from app.utils.feature_engineering import FeatureEngineer
from app.models.ml_models import VolatilityPredictor
from app.core.config import Settings

def setup_everything():
    """One-time setup: process data, engineer features, and train model"""
    
    print("🚀 Starting one-time setup process...")
    
    # Initialize components
    settings = Settings()
    data_preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    volatility_predictor = VolatilityPredictor()
    
    # Step 1: Load and preprocess raw data
    print("\n📊 Step 1: Loading and preprocessing raw data...")
    raw_data_path = "dataset.csv"
    
    if not os.path.exists(raw_data_path):
        print(f"❌ Error: {raw_data_path} not found!")
        return False
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    print(f"✅ Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Filter to top 5 cryptocurrencies for performance
    top_cryptos = ['Bitcoin', 'Ethereum', 'Litecoin', 'XRP', 'Cardano']
    df_filtered = df[df['crypto_name'].isin(top_cryptos)].copy()
    print(f"🎯 Filtered to top 5 cryptos: {len(df_filtered)} rows")
    
    # Preprocess data
    df_processed, quality_report = data_preprocessor.preprocess_pipeline(df_filtered)
    print(f"✅ Data preprocessing completed: {len(df_processed)} rows")
    
    # Step 2: Engineer features
    print("\n🔧 Step 2: Engineering features...")
    df_with_features = feature_engineer.feature_engineering_pipeline(df_processed)
    print(f"✅ Feature engineering completed: {df_with_features.shape[1]} features")
    
    # Step 3: Save processed data
    print("\n💾 Step 3: Saving processed data...")
    processed_data_path = "data/processed/processed_crypto_data.joblib"
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    joblib.dump(df_with_features, processed_data_path)
    print(f"✅ Processed data saved to: {processed_data_path}")
    
    # Step 4: Train model
    print("\n🎯 Step 4: Training model...")
    training_results = {}
    
    for crypto_name in df_with_features['crypto_name'].unique():
        print(f"  🔄 Training model for {crypto_name}...")
        crypto_data = df_with_features[df_with_features['crypto_name'] == crypto_name].copy()
        
        if len(crypto_data) >= 100:  # Minimum data points
            try:
                results = volatility_predictor.train(crypto_data, crypto_name)
                training_results[crypto_name] = results
                print(f"  ✅ {crypto_name} model trained successfully")
            except Exception as e:
                print(f"  ❌ Error training {crypto_name}: {e}")
                training_results[crypto_name] = {'error': str(e)}
        else:
            print(f"  ⚠️ Insufficient data for {crypto_name}")
    
    # Step 5: Save trained model
    print("\n💾 Step 5: Saving trained model...")
    model_path = "data/models/volatility_model.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    volatility_predictor.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Step 6: Save metadata
    print("\n📋 Step 6: Saving setup metadata...")
    metadata = {
        'setup_date': datetime.now().isoformat(),
        'data_shape': df_with_features.shape,
        'cryptocurrencies': list(df_with_features['crypto_name'].unique()),
        'feature_count': df_with_features.shape[1],
        'training_results': training_results,
        'data_range': {
            'start': df_with_features['date'].min().strftime('%Y-%m-%d') if 'date' in df_with_features.columns else 'N/A',
            'end': df_with_features['date'].max().strftime('%Y-%m-%d') if 'date' in df_with_features.columns else 'N/A'
        }
    }
    
    metadata_path = "data/processed/setup_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    print("\n🎉 Setup completed successfully!")
    print("📝 Summary:")
    print(f"   • Processed data: {df_with_features.shape[0]} rows, {df_with_features.shape[1]} features")
    print(f"   • Cryptocurrencies: {len(df_with_features['crypto_name'].unique())}")
    print(f"   • Models trained: {len([r for r in training_results.values() if 'error' not in r])}")
    print(f"   • Data saved to: {processed_data_path}")
    print(f"   • Model saved to: {model_path}")
    print("\n✅ Your application is now ready to run with pre-processed data!")
    
    return True

if __name__ == "__main__":
    success = setup_everything()
    if success:
        print("\n🚀 Run 'python app/main.py' to start the optimized application!")
    else:
        print("\n❌ Setup failed. Please check the errors above.")
