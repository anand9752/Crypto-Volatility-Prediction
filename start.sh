#!/bin/bash

# Create necessary directories
mkdir -p data/raw data/processed data/models logs reports

# Set environment variables
export PYTHONPATH=/opt/render/project/src

# Start the application with gunicorn
exec gunicorn app.main:app --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 120
