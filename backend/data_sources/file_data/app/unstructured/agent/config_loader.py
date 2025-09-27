import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

@dataclass
class DatabaseConfig:
    """Database configuration loaded from database."""
    host: str
    port: int
    database: str
    username: str
    password: str
    schema: str = "public"
    
    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Fallback to environment variables if database access fails."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5433)),
            database=os.getenv("DB_NAME", "document_db"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "1234"),
            schema=os.getenv("DB_SCHEMA", "public")
        )

def get_user_config_direct(user_id: str) -> Optional[DatabaseConfig]:
    """
    Directly access the database to get user configuration using the get_user_config function.
    This replaces the API call entirely.
    
    Args:
        user_id: The user ID to fetch configuration for
        
    Returns:
        DatabaseConfig object if successful, None otherwise
    """
    try:
        # Import here to avoid circular imports
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        # Connect to the main database where user configs are stored
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='main_db',
            user='postgres',
            password='1234'
        )
        
        # Updated query to get the latest configuration for the user
        select_query = """
        SELECT user_id, db_config, access_level, accessible_tables, created_at, updated_at
        FROM user_config
        WHERE user_id = %s AND is_latest = TRUE
        ORDER BY created_at DESC
        LIMIT 1;
        """
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(select_query, (user_id,))
            result = cursor.fetchone()
        
        conn.close()
        
        if result:
            config = dict(result)
            db_config_data = config.get('db_config', {})
            
            if db_config_data:
                logger.info(f"Found database config for user '{user_id}': {db_config_data.get('DB_NAME', 'unknown')}")
                return DatabaseConfig(
                    host=db_config_data.get("DB_HOST", "localhost"),
                    port=int(db_config_data.get("DB_PORT", 5433)),
                    database=db_config_data.get("DB_NAME", "document_db"),
                    username=db_config_data.get("DB_USER", "postgres"),
                    password=db_config_data.get("DB_PASSWORD", "1234"),
                    schema=db_config_data.get("schema", "public")
                )
            else:
                logger.warning(f"User '{user_id}' found but db_config is empty")
        else:
            logger.warning(f"No user config found in database for user_id: {user_id}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error accessing database directly for user config: {e}")
        return None

class ConfigLoader:
    """Handles loading configuration directly from database with fallback to environment variables."""
    
    def __init__(self):
        # No longer need API base URL since we're not using the API
        pass
    
    def load_database_config(self, user_id: str) -> DatabaseConfig:
        """
        Load database configuration directly from database.
        
        Args:
            user_id: The user ID to fetch configuration for
            
        Returns:
            DatabaseConfig object with database connection details
            
        Raises:
            Exception: If database access fails and no fallback environment variables are available
        """
        logger.info(f"🔍 Loading database config for user_id: {user_id}")
        
        # Try to load directly from database first
        try:
            direct_config = get_user_config_direct(user_id)
            if direct_config:
                logger.info(f"✅ Successfully loaded database config directly from database for user: {user_id}")
                logger.info(f"📊 Database: {direct_config.database} | Host: {direct_config.host}:{direct_config.port}")
                return direct_config
            else:
                logger.warning(f"⚠️ No user config found for user_id: {user_id}")
        except Exception as e:
            logger.warning(f"❌ Failed to load config directly from database for user {user_id}: {e}")
        
        # Fallback to environment variables
        logger.info("🔄 Falling back to environment variables for database configuration")
        env_config = DatabaseConfig.from_env()
        logger.info(f"📊 Environment config: {env_config.database} | Host: {env_config.host}:{env_config.port}")
        return env_config

# Global config loader instance
config_loader = ConfigLoader()

def get_database_config(user_id: Optional[str] = None) -> DatabaseConfig:
    """
    Get database configuration, trying direct database access first, then falling back to environment variables.
    
    Args:
        user_id: Optional user ID. If None, will use environment variables only.
        
    Returns:
        DatabaseConfig object
    """
    if user_id:
        return config_loader.load_database_config(user_id)
    else:
        logger.info("No user_id provided, using environment variables for database configuration")
        return DatabaseConfig.from_env() 