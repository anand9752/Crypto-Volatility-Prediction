# Optimized Dockerfile for Crypto Volatility Prediction Dashboard v3.0.0
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies (optimized)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ultra_fast_main.py .
COPY data/ ./data/
COPY app/ ./app/

# Ensure data directories exist with proper permissions
RUN mkdir -p data/models data/processed data/raw && \
    chmod -R 755 data/

# Create non-root user for security
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    chown -R user:user /app
USER user

# Expose port for the optimized application
EXPOSE 8000

# Health check for the optimized endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run the optimized application
CMD ["python", "ultra_fast_main.py"]
