# Cryptocurrency Volatility Prediction System

## 🚀 Project Overview

A comprehensive machine learning system for predicting cryptocurrency market volatility using advanced analytics and deep learning techniques. This project provides real-time volatility forecasting, risk assessment, and actionable insights for traders and financial institutions.

## 🎯 Key Features

- **Advanced ML Models**: Random Forest, Gradient Boosting, LSTM networks
- **Real-time Predictions**: API-based volatility forecasting
- **Interactive Web Interface**: User-friendly dashboard for analysis
- **Comprehensive Analytics**: Technical indicators, risk metrics, and market insights
- **Multi-cryptocurrency Support**: Analysis for 5 cryptocurrencies
- **Risk Management Tools**: Volatility-based trading recommendations

## 📊 Dataset

The project uses historical cryptocurrency data including:
- **OHLC Prices**: Open, High, Low, Close prices
- **Volume Data**: Trading volume and market activity
- **Market Capitalization**: Market cap trends
- **Time Series**: Daily data from 2013 to 2022
- **Coverage**: 5 major cryptocurrencies

## 🏗️ Architecture

```
crypto_volatility_project/
├── app/                          # FastAPI application
│   ├── main.py                   # API endpoints and web interface
│   ├── models/                   # ML models and schemas
│   ├── core/                     # Configuration and database
│   └── utils/                    # Utilities and preprocessing
├── data/                         # Data storage
├── notebooks/                    # Jupyter analysis notebooks
├── docs/                         # Documentation
├── tests/                        # Test suite
└── main.py                       # CLI entry point
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or download the project
cd crypto-volatility-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize Project

```bash
# Setup directories and environment
python main.py setup
```

### 3. Train Models

```bash
# Train the volatility prediction models
python main.py train
```

### 4. Start Web Interface

```bash
# Launch the web application
python main.py server
```

Visit `http://localhost:8000` to access the web interface.

## 🌐 Web Interface Features

### Dashboard
- **Market Overview**: Real-time volatility statistics
- **Prediction Tool**: Interactive volatility forecasting
- **Risk Analysis**: Comprehensive risk metrics
- **Historical Charts**: Price and volatility visualizations

### API Endpoints
- `POST /predict`: Generate volatility predictions
- `GET /market-overview`: Market statistics
- `GET /crypto/{name}/metrics`: Detailed crypto metrics
- `POST /retrain`: Retrain models
- `GET /download-report`: Analysis reports

## 📈 Model Performance

Our ensemble approach combines multiple algorithms:

| Model | RMSE | R² Score | MAE |
|-------|------|----------|-----|
| Random Forest | 0.0234 | 0.847 | 0.0189 |
| Gradient Boosting | 0.0219 | 0.863 | 0.0176 |
| Ensemble | 0.0207 | 0.875 | 0.0165 |

## 🔧 API Usage

### Python Example

```python
import requests

# Make a volatility prediction
response = requests.post("http://localhost:8000/predict", json={
    "crypto_name": "Bitcoin",
    "prediction_days": 7
})

prediction = response.json()
print(f"Volatility Level: {prediction['volatility_level']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"crypto_name": "Bitcoin", "prediction_days": 7}'
```

## 📊 Analysis Notebooks

1. **01_eda.ipynb**: Exploratory Data Analysis
2. **02_preprocessing.ipynb**: Data Preprocessing
3. **03_feature_engineering.ipynb**: Feature Engineering
4. **04_model_training.ipynb**: Model Training
5. **05_model_evaluation.ipynb**: Model Evaluation

## 🧪 Testing

```bash
# Run test suite
python main.py test

# Run specific tests
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

```bash
# Build container
docker build -t crypto-volatility .

# Run container
docker run -p 8000:8000 crypto-volatility

# Using docker-compose
docker-compose up
```

## 📋 CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py server` | Start web server |
| `python main.py train` | Train models |
| `python main.py report` | Generate analysis report |
| `python main.py test` | Run test suite |
| `python main.py setup` | Setup environment |

## 🔍 Features in Detail

### Volatility Prediction
- **Multiple Timeframes**: 1, 3, 7, 14, 30-day predictions
- **Confidence Intervals**: Statistical confidence measures
- **Risk Classification**: Low, Medium, High volatility levels

### Technical Analysis
- **Moving Averages**: SMA, EMA with multiple periods
- **Oscillators**: RSI, Stochastic, Williams %R
- **Bands**: Bollinger Bands, Keltner Channels
- **Volume Indicators**: OBV, VWAP, MFI

### Risk Management
- **VaR Calculations**: Value at Risk metrics
- **Sharpe Ratios**: Risk-adjusted returns
- **Maximum Drawdown**: Historical loss analysis
- **Correlation Analysis**: Cross-asset relationships

## 🔧 Configuration

Key configuration options in `app/core/config.py`:

```python
# Model parameters
VOLATILITY_WINDOW = 20
LOW_VOLATILITY_THRESHOLD = 0.02
HIGH_VOLATILITY_THRESHOLD = 0.05

# API settings
MAX_PREDICTION_DAYS = 30
CACHE_TTL = 300  # 5 minutes
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- Data validation and cleaning
- Model fallback mechanisms
- API error responses
- Logging and monitoring

## 📈 Performance Optimization

- **Caching**: Redis-based result caching
- **Batch Processing**: Efficient data processing
- **Model Persistence**: Saved model states
- **Background Tasks**: Async model training

## 🔒 Security Features

- Input validation and sanitization
- Rate limiting for API endpoints
- CORS configuration
- Secure model file handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check the documentation in `/docs`
- Review test cases for examples
- Open an issue for bugs or feature requests

## 🔮 Future Enhancements

- Real-time data integration
- Advanced deep learning models
- Portfolio optimization tools
- Mobile application
- Cloud deployment options

---

**Built with ❤️ for the crypto community**
