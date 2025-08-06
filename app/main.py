from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pickle
import joblib

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.schemas import (
    PredictionRequest, 
    PredictionResponse, 
    VolatilityMetrics,
    CryptoData,
    TrainingRequest
)
from app.models.ml_models import VolatilityPredictor
from app.core.config import get_settings
from app.utils.data_preprocessing import DataPreprocessor
from app.utils.feature_engineering import FeatureEngineer
from app.utils.model_utils import ModelManager

settings = get_settings()
app = FastAPI(
    title="Cryptocurrency Volatility Prediction API",
    description="ML-powered API for predicting cryptocurrency market volatility",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize global objects
model_manager = ModelManager()
data_preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()

@app.on_event("startup")
async def startup_event():
    """Initialize models and load data on startup"""
    try:
        # Load the trained model
        model_manager.load_model()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Could not load model - {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crypto Volatility Prediction</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            .card {
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            input, select {
                width: 100%;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                box-sizing: border-box;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                margin: 5px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .prediction-result {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin-top: 20px;
                border-radius: 0 8px 8px 0;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .metric-item {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            .metric-label {
                color: #666;
                font-size: 14px;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Cryptocurrency Volatility Prediction</h1>
                <p>Advanced ML-powered volatility forecasting for crypto markets</p>
            </div>
            
            <div class="card">
                <h2>📊 Quick Prediction</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="crypto">Cryptocurrency:</label>
                        <select id="crypto" name="crypto" required>
                            <option value="Bitcoin">Bitcoin</option>
                            <option value="Ethereum">Ethereum</option>
                            <option value="Litecoin">Litecoin</option>
                            <option value="XRP">XRP</option>
                            <option value="Cardano">Cardano</option>
                            <option value="Polkadot">Polkadot</option>
                            <option value="Chainlink">Chainlink</option>
                            <option value="Dogecoin">Dogecoin</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="days">Prediction Period (days):</label>
                        <select id="days" name="days" required>
                            <option value="1">1 Day</option>
                            <option value="3">3 Days</option>
                            <option value="7" selected>7 Days</option>
                            <option value="14">14 Days</option>
                            <option value="30">30 Days</option>
                        </select>
                    </div>
                    <button type="submit">🔮 Predict Volatility</button>
                </form>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Analyzing market data and generating predictions...</p>
                </div>
                
                <div id="predictionResult" style="display: none;">
                    <div class="prediction-result">
                        <h3>📈 Prediction Results</h3>
                        <div id="resultContent"></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Market Analysis</h2>
                <div class="metrics-grid" id="marketMetrics">
                    <!-- Market metrics will be loaded here -->
                </div>
                <div id="volatilityChart" style="height: 400px; margin-top: 20px;"></div>
            </div>
            
            <div class="card">
                <h2>🔧 Model Management</h2>
                <button onclick="retrainModel()">🔄 Retrain Model</button>
                <button onclick="downloadReport()">📋 Download Report</button>
                <button onclick="loadMarketData()">📊 Refresh Market Data</button>
                <div id="modelStatus" style="margin-top: 15px;"></div>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const crypto = document.getElementById('crypto').value;
                const days = document.getElementById('days').value;
                
                // Show loading
                document.getElementById('loading').style.display = 'block';
                document.getElementById('predictionResult').style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            crypto_name: crypto,
                            prediction_days: parseInt(days)
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayPredictionResult(result);
                    } else {
                        throw new Error(result.detail || 'Prediction failed');
                    }
                } catch (error) {
                    document.getElementById('resultContent').innerHTML = 
                        `<p style="color: red;">Error: ${error.message}</p>`;
                    document.getElementById('predictionResult').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            });
            
            function displayPredictionResult(result) {
                const resultContent = document.getElementById('resultContent');
                const volatilityLevel = result.volatility_level;
                const confidence = (result.confidence * 100).toFixed(1);
                
                let levelColor = '#28a745';
                if (volatilityLevel === 'Medium') levelColor = '#ffc107';
                if (volatilityLevel === 'High') levelColor = '#dc3545';
                
                resultContent.innerHTML = `
                    <div class="metrics-grid">
                        <div class="metric-item">
                            <div class="metric-value" style="color: ${levelColor};">${volatilityLevel}</div>
                            <div class="metric-label">Volatility Level</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${confidence}%</div>
                            <div class="metric-label">Confidence</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">${result.predicted_volatility.toFixed(4)}</div>
                            <div class="metric-label">Volatility Score</div>
                        </div>
                    </div>
                    <p><strong>Recommendation:</strong> ${result.recommendation}</p>
                `;
                
                document.getElementById('predictionResult').style.display = 'block';
            }
            
            async function retrainModel() {
                document.getElementById('modelStatus').innerHTML = 'Retraining model...';
                try {
                    const response = await fetch('/retrain', { method: 'POST' });
                    const result = await response.json();
                    document.getElementById('modelStatus').innerHTML = 
                        `<p style="color: green;">Model retrained successfully!</p>`;
                } catch (error) {
                    document.getElementById('modelStatus').innerHTML = 
                        `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
            
            async function downloadReport() {
                window.open('/download-report', '_blank');
            }
            
            async function loadMarketData() {
                try {
                    const response = await fetch('/market-overview');
                    const data = await response.json();
                    displayMarketMetrics(data);
                } catch (error) {
                    console.error('Error loading market data:', error);
                }
            }
            
            function displayMarketMetrics(data) {
                const metricsContainer = document.getElementById('marketMetrics');
                metricsContainer.innerHTML = `
                    <div class="metric-item">
                        <div class="metric-value">${data.total_cryptos}</div>
                        <div class="metric-label">Tracked Cryptocurrencies</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.avg_volatility.toFixed(3)}</div>
                        <div class="metric-label">Average Market Volatility</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.high_volatility_count}</div>
                        <div class="metric-label">High Volatility Assets</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">${data.data_points.toLocaleString()}</div>
                        <div class="metric-label">Total Data Points</div>
                    </div>
                `;
            }
            
            // Load initial market data
            loadMarketData();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_volatility(request: PredictionRequest):
    """Predict cryptocurrency volatility"""
    try:
        # Load latest data for the cryptocurrency
        data = model_manager.get_latest_data(request.crypto_name)
        
        if data is None or len(data) < 30:
            raise HTTPException(
                status_code=404, 
                detail=f"Insufficient data for {request.crypto_name}"
            )
        
        # Generate prediction
        prediction = model_manager.predict(data, request.prediction_days)
        
        return PredictionResponse(
            crypto_name=request.crypto_name,
            prediction_days=request.prediction_days,
            predicted_volatility=prediction['volatility'],
            volatility_level=prediction['level'],
            confidence=prediction['confidence'],
            recommendation=prediction['recommendation'],
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-overview")
async def get_market_overview():
    """Get market overview statistics"""
    try:
        stats = model_manager.get_market_overview()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crypto/{crypto_name}/metrics")
async def get_crypto_metrics(crypto_name: str):
    """Get detailed metrics for a specific cryptocurrency"""
    try:
        metrics = model_manager.get_crypto_metrics(crypto_name)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the volatility prediction model"""
    background_tasks.add_task(model_manager.retrain_model)
    return {"message": "Model retraining started in background"}

@app.get("/download-report")
async def download_report():
    """Download analysis report"""
    try:
        report_path = model_manager.generate_report()
        return FileResponse(
            path=report_path,
            media_type='application/pdf',
            filename="volatility_analysis_report.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model_manager.is_model_loaded()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
