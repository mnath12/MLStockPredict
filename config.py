"""
Configuration module for MLStockPredict

Loads API keys and configuration from environment variables.
Place your API keys in a .env file in the project root.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# API Keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
FRED_API_KEY = os.getenv('FRED_API_KEY', '')

# Environment settings
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validate required keys in production
def validate_config():
    """Validate that required API keys are present in production."""
    if ENVIRONMENT == 'production':
        required_keys = {
            'ALPACA_API_KEY': ALPACA_API_KEY,
            'ALPACA_SECRET_KEY': ALPACA_SECRET_KEY,
            'POLYGON_API_KEY': POLYGON_API_KEY,
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(
                f"Missing required API keys in production: {', '.join(missing_keys)}\n"
                f"Please create a .env file with your API keys. See env.example for format."
            )
    
    return True

# Auto-validate on import (can be disabled if needed)
if __name__ == '__main__':
    validate_config()
    print("✓ Configuration loaded successfully")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Log Level: {LOG_LEVEL}")
    print(f"  Alpaca API Key: {'✓' if ALPACA_API_KEY else '✗'}")
    print(f"  Alpaca Secret: {'✓' if ALPACA_SECRET_KEY else '✗'}")
    print(f"  Polygon API Key: {'✓' if POLYGON_API_KEY else '✗'}")
    print(f"  FRED API Key: {'✓' if FRED_API_KEY else '✗'}")

