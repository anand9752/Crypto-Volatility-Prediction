from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class VolatilityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class CryptoData(BaseModel):
    """Schema for cryptocurrency data input"""
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    marketCap: float = Field(..., description="Market capitalization")
    crypto_name: str = Field(..., description="Cryptocurrency name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

class PredictionRequest(BaseModel):
    """Schema for volatility prediction request"""
    crypto_name: str = Field(..., description="Name of the cryptocurrency")
    prediction_days: int = Field(default=7, ge=1, le=30, description="Number of days to predict")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")

class PredictionResponse(BaseModel):
    """Schema for volatility prediction response"""
    crypto_name: str
    prediction_days: int
    predicted_volatility: float = Field(..., description="Predicted volatility score")
    volatility_level: VolatilityLevel = Field(..., description="Volatility level classification")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence score")
    recommendation: str = Field(..., description="Trading recommendation")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "crypto_name": "Bitcoin",
                "prediction_days": 7,
                "predicted_volatility": 0.0234,
                "volatility_level": "Medium",
                "confidence": 0.87,
                "recommendation": "Moderate risk - consider position sizing",
                "timestamp": "2025-08-06T10:30:00"
            }
        }

class VolatilityMetrics(BaseModel):
    """Schema for volatility metrics"""
    daily_volatility: float
    weekly_volatility: float
    monthly_volatility: float
    volatility_trend: str
    risk_score: float
    sharpe_ratio: Optional[float] = None

class TrainingRequest(BaseModel):
    """Schema for model training request"""
    retrain_from_scratch: bool = Field(default=False, description="Whether to retrain from scratch")
    include_recent_data: bool = Field(default=True, description="Include recent data in training")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.3, description="Validation data split ratio")

class TrainingResponse(BaseModel):
    """Schema for model training response"""
    status: str
    training_duration: float
    model_accuracy: float
    validation_score: float
    model_version: str
    timestamp: datetime

class MarketOverview(BaseModel):
    """Schema for market overview"""
    total_cryptos: int
    avg_volatility: float
    high_volatility_count: int
    medium_volatility_count: int
    low_volatility_count: int
    data_points: int
    last_updated: datetime

class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: float
    mae: float
    model_version: str
    training_date: datetime

class HealthCheck(BaseModel):
    """Schema for health check response"""
    status: str
    timestamp: datetime
    model_loaded: bool
    data_available: bool
    api_version: str
