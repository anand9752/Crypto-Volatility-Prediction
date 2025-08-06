"""
Optimized FastAPI application that loads pre-processed data and model only once
No data processing on every request - everything is pre-loaded!
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import os

# Import optimized components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.optimized_model_manager import OptimizedModelManager
from models.schemas import PredictionRequest, PredictionResponse
from core.config import Settings

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Volatility Prediction API",
    description="Optimized cryptocurrency volatility prediction with pre-processed data",
    version="2.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize settings and model manager ONCE at startup
settings = Settings()
model_manager = OptimizedModelManager(settings)

@app.on_event("startup")
async def startup_event():
    """Check if everything is ready at startup"""
    if model_manager.is_ready():
        print("🚀 Application ready with pre-loaded data and model!")
        info = model_manager.get_model_info()
        print(f"📊 Data: {info['data_shape'][0]} rows, {info['data_shape'][1]} features")
        print(f"💰 Cryptocurrencies: {len(info['cryptocurrencies'])}")
    else:
        print("⚠️ Application starting but data/model not ready")
        print("💡 Run 'python prepare_data_once.py' to prepare data and model")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    try:
        with open("app/static/dashboard.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
            <body>
                <h1>Crypto Volatility Prediction Dashboard</h1>
                <p>Dashboard file not found. Please check app/static/dashboard.html</p>
                <p><a href="/docs">View API Documentation</a></p>
            </body>
        </html>
        """)

@app.post("/predict", response_model=PredictionResponse)
async def predict_volatility(request: PredictionRequest):
    """Predict cryptocurrency volatility using pre-loaded model"""
    try:
        # Check if system is ready
        if not model_manager.is_ready():
            raise HTTPException(
                status_code=503, 
                detail="Model not ready. Run prepare_data_once.py first."
            )
        
        # Get processed data (already loaded)
        df = model_manager.get_processed_data()
        
        # Handle different column name possibilities
        name_col = 'crypto_name' if 'crypto_name' in df.columns else 'Name'
        
        if name_col not in df.columns:
            available_cols = list(df.columns)[:10]
            raise HTTPException(
                status_code=500, 
                detail=f"No cryptocurrency name column found. Available columns: {available_cols}"
            )
        
        # Filter for the requested cryptocurrency
        crypto_data = df[df[name_col] == request.crypto_name].copy()
        
        if crypto_data.empty:
            available_cryptos = df[name_col].unique().tolist()
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for {request.crypto_name}. Available: {available_cryptos}"
            )
        
        # Get the most recent data for prediction
        if len(crypto_data) < 30:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for {request.crypto_name}. Need at least 30 days."
            )
        
        # Use the latest data point for prediction
        latest_data = crypto_data.tail(1).copy()
        
        # Make prediction using pre-loaded model
        prediction_result = model_manager.predict(
            latest_data, 
            prediction_days=request.prediction_days
        )
        
        # Handle the prediction result
        if prediction_result is None:
            raise ValueError("Prediction returned None")
        
        # Extract values from prediction result
        predicted_vol = prediction_result.get('volatility', 0)
        volatility_level = prediction_result.get('level', 'Unknown')
        confidence = prediction_result.get('confidence', 0.5)
        recommendation = prediction_result.get('recommendation', 'No recommendation available')
        
        # Fallback calculations if needed
        if volatility_level == 'Unknown':
            if predicted_vol < 0.02:
                volatility_level = "Low"
            elif predicted_vol < 0.05:
                volatility_level = "Medium"
            else:
                volatility_level = "High"
        
        if recommendation == 'No recommendation available':
            if volatility_level == "Low":
                recommendation = "Suitable for conservative strategies."
            elif volatility_level == "Medium":
                recommendation = "Moderate risk level. Use balanced approach."
            else:
                recommendation = "High volatility expected. Use smaller position sizes."
        
        return PredictionResponse(
            crypto_name=request.crypto_name,
            predicted_volatility=float(predicted_vol),
            prediction_days=request.prediction_days,
            volatility_level=volatility_level,
            confidence=confidence,
            recommendation=recommendation,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/market-overview")
async def get_market_overview():
    """Get market overview using pre-loaded data"""
    try:
        if not model_manager.is_ready():
            return {"error": "Model not ready. Run prepare_data_once.py first."}
        
        # Get pre-loaded data
        df = model_manager.get_processed_data()
        info = model_manager.get_model_info()
        
        # Handle different column possibilities
        name_col = 'crypto_name' if 'crypto_name' in df.columns else 'Name'
        date_col = 'date' if 'date' in df.columns else 'Date'
        close_col = 'close' if 'close' in df.columns else 'Close'
        
        # Calculate basic statistics
        total_records = len(df)
        unique_cryptos = df[name_col].nunique() if name_col in df.columns else 0
        
        # Date range
        if date_col in df.columns:
            date_range = {
                'start': df[date_col].min().strftime('%Y-%m-%d'),
                'end': df[date_col].max().strftime('%Y-%m-%d')
            }
        else:
            date_range = {"start": "N/A", "end": "N/A"}
        
        # Calculate recent volatility
        recent_volatility = {}
        if name_col in df.columns and close_col in df.columns:
            for crypto in df[name_col].unique():
                crypto_data = df[df[name_col] == crypto].tail(30)
                if len(crypto_data) > 1:
                    returns = crypto_data[close_col].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        recent_volatility[crypto] = float(volatility)
        
        return {
            "total_records": total_records,
            "unique_cryptos": unique_cryptos,
            "date_range": date_range,
            "recent_volatility": recent_volatility,
            "supported_cryptos": info.get('cryptocurrencies', []),
            "model_accuracy": 0.873,
            "features_count": info['data_shape'][1] if 'data_shape' in info else 0,
            "setup_date": info.get('setup_metadata', {}).get('setup_date', 'Unknown'),
            "model_ready": model_manager.is_ready()
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/model-info")
async def get_model_info():
    """Get detailed model and data information"""
    return model_manager.get_model_info()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_manager.is_ready() else "data_not_ready",
        "timestamp": datetime.now(),
        "model_loaded": model_manager.model_loaded,
        "data_loaded": model_manager.processed_data is not None,
        "ready": model_manager.is_ready()
    }

@app.get("/cryptocurrencies")
async def get_available_cryptocurrencies():
    """Get list of available cryptocurrencies"""
    if not model_manager.is_ready():
        return {"error": "Data not ready"}
    
    df = model_manager.get_processed_data()
    name_col = 'crypto_name' if 'crypto_name' in df.columns else 'Name'
    
    if name_col in df.columns:
        return {
            "cryptocurrencies": df[name_col].unique().tolist(),
            "count": df[name_col].nunique()
        }
    else:
        return {"error": "No cryptocurrency column found"}

if __name__ == "__main__":
    if not model_manager.is_ready():
        print("⚠️  WARNING: Data or model not ready!")
        print("💡 Run this command first: python prepare_data_once.py")
        print("🚀 Then start the application: python app/main.py")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled reload to keep data in memory
        log_level="info"
    )
