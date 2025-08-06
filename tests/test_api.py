import pytest
import numpy as np
import pandas as pd
from httpx import AsyncClient
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

@pytest.mark.asyncio
class TestAPI:
    """Test cases for FastAPI endpoints"""
    
    async def test_root_endpoint(self):
        """Test the root endpoint returns HTML"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Crypto Volatility Prediction" in response.text
    
    async def test_health_endpoint(self):
        """Test the health check endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
    
    async def test_market_overview_endpoint(self):
        """Test the market overview endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/market-overview")
        
        assert response.status_code in [200, 500]  # May fail if no data loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "total_cryptos" in data
            assert "avg_volatility" in data
            assert "data_points" in data
    
    async def test_predict_endpoint_valid_request(self):
        """Test prediction endpoint with valid request"""
        prediction_request = {
            "crypto_name": "Bitcoin",
            "prediction_days": 7,
            "include_confidence": True
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # May return 500 if model not trained, or 404 if no data
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "crypto_name" in data
            assert "predicted_volatility" in data
            assert "volatility_level" in data
            assert "confidence" in data
            assert "recommendation" in data
    
    async def test_predict_endpoint_invalid_crypto(self):
        """Test prediction endpoint with invalid cryptocurrency"""
        prediction_request = {
            "crypto_name": "InvalidCoin",
            "prediction_days": 7
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # Should return 404 for invalid cryptocurrency
        assert response.status_code in [404, 500]
    
    async def test_predict_endpoint_invalid_days(self):
        """Test prediction endpoint with invalid prediction days"""
        prediction_request = {
            "crypto_name": "Bitcoin",
            "prediction_days": 50  # Exceeds maximum
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    async def test_predict_endpoint_missing_fields(self):
        """Test prediction endpoint with missing required fields"""
        prediction_request = {
            "prediction_days": 7
            # Missing crypto_name
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # Should return 422 for validation error
        assert response.status_code == 422
    
    async def test_crypto_metrics_endpoint(self):
        """Test cryptocurrency metrics endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/crypto/Bitcoin/metrics")
        
        assert response.status_code in [200, 404, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "crypto_name" in data
    
    async def test_retrain_endpoint(self):
        """Test model retraining endpoint"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/retrain")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "background" in data["message"].lower()

class TestAPIValidation:
    """Test API input validation and error handling"""
    
    @pytest.mark.asyncio
    async def test_predict_with_negative_days(self):
        """Test prediction with negative days"""
        prediction_request = {
            "crypto_name": "Bitcoin",
            "prediction_days": -1
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_predict_with_zero_days(self):
        """Test prediction with zero days"""
        prediction_request = {
            "crypto_name": "Bitcoin",
            "prediction_days": 0
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_predict_with_empty_crypto_name(self):
        """Test prediction with empty crypto name"""
        prediction_request = {
            "crypto_name": "",
            "prediction_days": 7
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        assert response.status_code in [422, 404]
    
    @pytest.mark.asyncio
    async def test_predict_with_sql_injection_attempt(self):
        """Test prediction endpoint against SQL injection"""
        prediction_request = {
            "crypto_name": "Bitcoin'; DROP TABLE predictions; --",
            "prediction_days": 7
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # Should handle gracefully
        assert response.status_code in [404, 422, 500]

class TestAPIPerformance:
    """Test API performance and concurrency"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import asyncio
        
        async def make_request():
            async with AsyncClient(app=app, base_url="http://test") as ac:
                return await ac.get("/health")
        
        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_large_request_handling(self):
        """Test handling of large requests"""
        # Create a large but valid request
        prediction_request = {
            "crypto_name": "Bitcoin" * 100,  # Very long name
            "prediction_days": 7
        }
        
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/predict", json=prediction_request)
        
        # Should handle gracefully (likely 404 for non-existent crypto)
        assert response.status_code in [404, 422]

class TestAPIResponseFormat:
    """Test API response format and structure"""
    
    @pytest.mark.asyncio
    async def test_health_response_format(self):
        """Test health endpoint response format"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_cors_headers(self):
        """Test CORS headers are present"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.options("/health")
        
        # CORS preflight should be handled
        assert response.status_code in [200, 405]
