"""
Database configuration module
All configuration is loaded from .env file
"""
import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def load_config():
    """
    Load database configuration from environment variables (.env file)

    Required environment variables:
    - DB_HOST: Database host (default: localhost)
    - DB_PORT: Database port (default: 5432)
    - DB_USER: Database user (default: postgres)
    - DB_PASSWORD: Database password (required for production)
    - DB_NAME: Database name (default: petrolimex_ai)

    Returns:
        dict: Database configuration
    """
    config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'user': os.getenv('DB_USER') or os.getenv('DB_USERNAME', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres'),
        'database': os.getenv('DB_NAME') or os.getenv('DB_DATABASE', 'petrolimex_ai'),
    }

    # Validate required fields in production
    environment = os.getenv('ENVIRONMENT', 'development')
    if environment == 'production' and config['password'] == 'postgres':
        raise ValueError(
            "Production environment requires DB_PASSWORD to be set in .env file!"
        )

    return config


# Export config as module-level variable
config = load_config()

