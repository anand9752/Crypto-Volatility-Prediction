import sqlite3
import pandas as pd
import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from app.core.config import get_settings

settings = get_settings()

class DatabaseManager:
    """Database manager for storing and retrieving cryptocurrency data and predictions"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(settings.DATA_PATH, "crypto_volatility.db")
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Crypto data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    market_cap REAL,
                    timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(crypto_name, date)
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    crypto_name TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    predicted_volatility REAL,
                    volatility_level TEXT,
                    confidence REAL,
                    prediction_days INTEGER,
                    model_version TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    training_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Model training history
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT NOT NULL,
                    training_start TIMESTAMP,
                    training_end TIMESTAMP,
                    data_points INTEGER,
                    accuracy REAL,
                    r2_score REAL,
                    mse REAL,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def insert_crypto_data(self, data: pd.DataFrame):
        """Insert cryptocurrency data into database"""
        with sqlite3.connect(self.db_path) as conn:
            # Prepare data
            data_to_insert = data.copy()
            if 'Unnamed: 0' in data_to_insert.columns:
                data_to_insert = data_to_insert.drop('Unnamed: 0', axis=1)
            
            # Rename columns to match database schema
            column_mapping = {
                'marketCap': 'market_cap'
            }
            data_to_insert = data_to_insert.rename(columns=column_mapping)
            
            # Insert data
            data_to_insert.to_sql('crypto_data', conn, if_exists='replace', index=False)
    
    def get_crypto_data(self, crypto_name: Optional[str] = None, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """Retrieve cryptocurrency data from database"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM crypto_data WHERE 1=1"
            params = []
            
            if crypto_name:
                query += " AND crypto_name = ?"
                params.append(crypto_name)
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY crypto_name, date"
            
            return pd.read_sql_query(query, conn, params=params)
    
    def insert_prediction(self, crypto_name: str, prediction_data: Dict[str, Any]):
        """Insert prediction result into database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions 
                (crypto_name, prediction_date, predicted_volatility, volatility_level, 
                 confidence, prediction_days, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                crypto_name,
                datetime.now().isoformat(),
                prediction_data.get('volatility'),
                prediction_data.get('level'),
                prediction_data.get('confidence'),
                prediction_data.get('prediction_days', 7),
                prediction_data.get('model_version', '1.0')
            ))
            
            conn.commit()
    
    def get_predictions(self, crypto_name: Optional[str] = None, 
                       limit: int = 100) -> pd.DataFrame:
        """Retrieve prediction history"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM predictions 
                WHERE 1=1
            """
            params = []
            
            if crypto_name:
                query += " AND crypto_name = ?"
                params.append(crypto_name)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            return pd.read_sql_query(query, conn, params=params)
    
    def insert_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Insert model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for metric_name, metric_value in metrics.items():
                cursor.execute("""
                    INSERT INTO model_metrics 
                    (model_name, metric_name, metric_value, training_date)
                    VALUES (?, ?, ?, ?)
                """, (model_name, metric_name, metric_value, datetime.now().isoformat()))
            
            conn.commit()
    
    def get_model_metrics(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Retrieve model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM model_metrics WHERE 1=1"
            params = []
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY created_at DESC"
            
            return pd.read_sql_query(query, conn, params=params)
    
    def insert_training_record(self, training_data: Dict[str, Any]):
        """Insert training session record"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO training_history 
                (model_version, training_start, training_end, data_points, 
                 accuracy, r2_score, mse, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                training_data.get('model_version'),
                training_data.get('training_start'),
                training_data.get('training_end'),
                training_data.get('data_points'),
                training_data.get('accuracy'),
                training_data.get('r2_score'),
                training_data.get('mse'),
                json.dumps(training_data.get('config', {}))
            ))
            
            conn.commit()
    
    def get_training_history(self, limit: int = 50) -> pd.DataFrame:
        """Retrieve model training history"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM training_history 
                ORDER BY created_at DESC 
                LIMIT ?
            """
            return pd.read_sql_query(query, conn, params=[limit])
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get market summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total cryptocurrencies
            cursor.execute("SELECT COUNT(DISTINCT crypto_name) as total_cryptos FROM crypto_data")
            total_cryptos = cursor.fetchone()[0]
            
            # Total data points
            cursor.execute("SELECT COUNT(*) as total_records FROM crypto_data")
            total_records = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(date) as min_date, MAX(date) as max_date FROM crypto_data")
            date_range = cursor.fetchone()
            
            # Recent predictions count
            cursor.execute("""
                SELECT COUNT(*) as recent_predictions 
                FROM predictions 
                WHERE created_at >= datetime('now', '-7 days')
            """)
            recent_predictions = cursor.fetchone()[0]
            
            return {
                'total_cryptos': total_cryptos or 0,
                'total_records': total_records or 0,
                'min_date': date_range[0] if date_range else None,
                'max_date': date_range[1] if date_range else None,
                'recent_predictions': recent_predictions or 0,
                'last_updated': datetime.now().isoformat()
            }
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Clean up old prediction data"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            cursor.execute("""
                DELETE FROM predictions 
                WHERE created_at < ?
            """, (cutoff_date.isoformat(),))
            
            conn.commit()
            
            return cursor.rowcount
