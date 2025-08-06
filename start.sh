#!/bin/bash

# Create necessary directories
mkdir -p data/raw data/processed data/models logs reports

# Set environment variables
export PYTHONPATH=/opt/render/project/src
export ENVIRONMENT=production

# Print startup info
echo "🚀 Starting Cryptocurrency Volatility Prediction API..."
echo "🌐 Environment: $ENVIRONMENT"
echo "📊 Port: $PORT"

# Start the application with gunicorn directly (NOT through main.py)
exec gunicorn app.main:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --preload \
    --log-level info
