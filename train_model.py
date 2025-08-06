#!/usr/bin/env python3
"""
Script to train the cryptocurrency volatility prediction model
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.model_utils import ModelManager
from app.core.config import get_settings

def main():
    """Train the volatility prediction model"""
    print("🚀 Starting Cryptocurrency Volatility Model Training")
    print("=" * 60)
    
    try:
        # Initialize model manager
        print("📊 Initializing model manager...")
        model_manager = ModelManager()
        
        # Check if data is loaded
        if model_manager.processed_data is None:
            print("❌ No data available for training. Please check the dataset file.")
            return False
        
        print(f"✅ Data loaded: {model_manager.processed_data.shape[0]} rows, {model_manager.processed_data.shape[1]} features")
        print(f"📈 Cryptocurrencies available: {model_manager.processed_data['crypto_name'].nunique()}")
        
        # Start training
        print("\n🔄 Starting model training process...")
        training_start = datetime.now()
        
        training_results = model_manager.train_model(retrain=False)
        
        training_end = datetime.now()
        training_duration = (training_end - training_start).total_seconds()
        
        # Display results
        print("\n" + "=" * 60)
        print("✅ MODEL TRAINING COMPLETED!")
        print("=" * 60)
        
        if training_results['status'] == 'success':
            print(f"⏱️  Training Duration: {training_duration:.2f} seconds")
            print(f"🏗️  Model Version: {training_results['model_version']}")
            print(f"💰 Cryptocurrencies Trained: {training_results['cryptocurrencies_trained']}")
            
            # Display individual crypto results
            print(f"\n📊 Training Results Summary:")
            print("-" * 40)
            
            successful_trainings = 0
            failed_trainings = 0
            
            for crypto_name, results in training_results['training_results'].items():
                if isinstance(results, dict) and 'error' not in results:
                    successful_trainings += 1
                    # Get best performing model for this crypto
                    best_model = max(results.items(), key=lambda x: x[1].get('r2', 0) if isinstance(x[1], dict) else 0)
                    if isinstance(best_model[1], dict):
                        print(f"✅ {crypto_name:15} | Best: {best_model[0]:15} | R²: {best_model[1].get('r2', 0):.4f} | MSE: {best_model[1].get('mse', 0):.6f}")
                    else:
                        print(f"✅ {crypto_name:15} | Trained successfully")
                else:
                    failed_trainings += 1
                    error_msg = results.get('error', 'Unknown error') if isinstance(results, dict) else str(results)
                    print(f"❌ {crypto_name:15} | Error: {error_msg}")
            
            print("-" * 40)
            print(f"📈 Successful: {successful_trainings} | ❌ Failed: {failed_trainings}")
            
            # Verify model file exists
            settings = get_settings()
            if os.path.exists(settings.MODEL_FILE):
                print(f"✅ Model saved to: {settings.MODEL_FILE}")
            else:
                print(f"⚠️  Warning: Model file not found at expected location: {settings.MODEL_FILE}")
            
            # Test model loading
            print(f"\n🔍 Testing model loading...")
            try:
                test_manager = ModelManager()
                test_manager.load_model()
                if test_manager.is_model_loaded():
                    print("✅ Model loads successfully!")
                else:
                    print("❌ Model failed to load properly")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
            
            print(f"\n🎉 Training completed successfully!")
            print(f"💡 You can now run the application and make predictions!")
            
            return True
            
        else:
            print(f"❌ Training failed with status: {training_results['status']}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print(f"\n🚀 Next steps:")
        print(f"1. Run the application: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print(f"2. Open http://localhost:8000 in your browser")
        print(f"3. Start making volatility predictions!")
        sys.exit(0)
    else:
        print(f"\n💥 Training failed. Please check the errors above and try again.")
        sys.exit(1)
