#!/usr/bin/env python3
"""
Cryptocurrency Volatility Prediction System
Main entry point for the application
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_api_server():
    """Run the FastAPI server"""
    import uvicorn
    from app.main import app
    
    print("🚀 Starting Cryptocurrency Volatility Prediction API...")
    print("📊 API Documentation available at: http://localhost:8000/docs")
    print("🌐 Web Interface available at: http://localhost:8000")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

def train_model():
    """Train the volatility prediction model"""
    from app.utils.model_utils import ModelManager
    
    print("🤖 Starting model training...")
    
    model_manager = ModelManager()
    results = model_manager.train_model()
    
    print(f"✅ Training completed successfully!")
    print(f"📊 Training results: {results}")

def generate_report():
    """Generate analysis report"""
    from app.utils.model_utils import ModelManager
    
    print("📋 Generating analysis report...")
    
    model_manager = ModelManager()
    report_path = model_manager.generate_report()
    
    print(f"✅ Report generated: {report_path}")

def run_tests():
    """Run test suite"""
    import pytest
    
    print("🧪 Running test suite...")
    
    test_dir = project_root / "tests"
    if test_dir.exists():
        pytest.main([str(test_dir), "-v"])
    else:
        print("❌ Tests directory not found")

def setup_environment():
    """Setup the development environment"""
    print("⚙️ Setting up development environment...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/models",
        "logs",
        "reports"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("✅ Environment setup completed!")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Volatility Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server          # Start the API server
  python main.py train           # Train the model
  python main.py report          # Generate analysis report
  python main.py test            # Run tests
  python main.py setup           # Setup environment
        """
    )
    
    parser.add_argument(
        "command",
        choices=["server", "train", "report", "test", "setup"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for API server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    args = parser.parse_args()
    
    if args.command == "server":
        if args.host != "0.0.0.0" or args.port != 8000:
            import uvicorn
            from app.main import app
            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                reload=args.reload
            )
        else:
            run_api_server()
    
    elif args.command == "train":
        train_model()
    
    elif args.command == "report":
        generate_report()
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "setup":
        setup_environment()

if __name__ == "__main__":
    main()
