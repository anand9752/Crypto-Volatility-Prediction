"""
Simple app launcher for production deployment
This file is used by Render/Heroku/etc. for direct app launching
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set production environment
os.environ.setdefault("ENVIRONMENT", "production")

# Import the FastAPI app
from app.main import app

# This is what gunicorn will import
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
