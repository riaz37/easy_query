from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, ArgumentError, NoSuchModuleError
from langchain.sql_database import SQLDatabase
import logging
from typing import Optional, List
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_database_url(database_url: str) -> dict:
    """
    Validate database URL format and extract connection details.
    
    Args:
        database_url (str): The database connection URL
        
    Returns:
        dict: Validation result with details about the URL
        
    Raises:
        ValueError: If URL format is invalid
    """
    # Common database URL patterns
    patterns = {
        'mssql': r'mssql\+pyodbc://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)',
        'postgresql': r'postgresql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)',
        'mysql': r'mysql\+pymysql://([^:]+):([^@]+)@([^:]+):(\d+)/([^?]+)',
        'sqlite': r'sqlite:///(.+)'
    }
    
    for db_type, pattern in patterns.items():
        match = re.match(pattern, database_url)
        if match:
            if db_type == 'sqlite':
                return {
                    'valid': True,
                    'type': db_type,
                    'database': match.group(1),
                    'message': f"Valid {db_type} URL format detected"
                }
            else:
                username, password, host, port, database = match.groups()
                return {
                    'valid': True,
                    'type': db_type,
                    'username': username,
                    'host': host,
                    'port': port,
                    'database': database,
                    'message': f"Valid {db_type} URL format detected - Host: {host}:{port}, Database: {database}"
                }
    
    return {
        'valid': False,
        'message': "Invalid database URL format. Expected formats:\n" +
                  "- MSSQL: mssql+pyodbc://username:password@host:port/database?driver=...\n" +
                  "- PostgreSQL: postgresql://username:password@host:port/database\n" +
                  "- MySQL: mysql+pymysql://username:password@host:port/database\n" +
                  "- SQLite: sqlite:///path/to/database.db"
    }

def generate_table_info(
    database_url: str,
    output_file_path: Optional[str] = None,
    include_tables: Optional[List[str]] = None,
    sample_rows_in_table_info: int = 2,
    max_string_length: int = 100,
    pool_size: int = 5,
    max_overflow: int = 10
) -> str:
    """
    Generate table information from a database and save it to a file.
    
    Args:
        database_url (str): The database connection URL
        output_file_path (str, optional): Path to save the table info file. 
                                        Defaults to 'table_info_new.txt' in current directory
        include_tables (List[str], optional): List of specific tables to include. 
                                            If None, includes all tables
        sample_rows_in_table_info (int): Number of sample rows to include in table info
        max_string_length (int): Maximum length for string values in samples
        pool_size (int): Database connection pool size
        max_overflow (int): Maximum overflow connections
    
    Returns:
        str: The table information as a string
        
    Raises:
        ValueError: If database URL is invalid or connection fails
        ConnectionError: If network connectivity issues occur
        ImportError: If required database drivers are missing
    """
    
    # Validate database URL format first
    validation = validate_database_url(database_url)
    if not validation['valid']:
        raise ValueError(f"Database URL validation failed: {validation['message']}")
    
    logger.info(f"Database URL validation passed: {validation['message']}")
    
    # Set default output file path
    if output_file_path is None:
        output_file_path = Path(__file__).parent / "table_info_new.txt"
    else:
        output_file_path = Path(output_file_path)
    
    # Ensure output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    engine = None
    db = None
    
    try:
        logger.info(f"Creating database engine for URL: {database_url[:20]}...")
        
        # Create SQLAlchemy engine with connection pooling (no timeout for long operations)
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        logger.info("Creating SQLDatabase instance...")
        
        # Create SQLDatabase instance
        db_kwargs = {
            'sample_rows_in_table_info': sample_rows_in_table_info,
            'max_string_length': max_string_length
        }
        
        # Add include_tables only if specified
        if include_tables:
            db_kwargs['include_tables'] = include_tables
            logger.info(f"Including specific tables: {include_tables}")
        
        db = SQLDatabase(engine, **db_kwargs)
        
        logger.info("Extracting table information...")
        
        # Get table information
        table_info = db.get_table_info()
        
        if not table_info or table_info.strip() == "":
            raise ValueError("No table information could be extracted from the database. This might indicate:\n" +
                           "1. The database is empty (no tables exist)\n" +
                           "2. The user lacks permissions to view table information\n" +
                           "3. The database connection succeeded but no schema information is available")
        
        logger.info(f"Saving table info to: {output_file_path}")
        
        # Save table info to file
        output_file_path.write_text(table_info, encoding='utf-8')
        
        logger.info(f"Successfully generated table info ({len(table_info)} characters)")
        
        return table_info
        
    except (ArgumentError, NoSuchModuleError, OperationalError, Exception) as e:
        logger.error(f"Database connection error: {str(e)}")
        raise ValueError("Please check databaseurl")
    
    finally:
        # Clean up resources
        if engine:
            try:
                engine.dispose()
                logger.info("Database engine disposed")
            except Exception as e:
                logger.warning(f"Error disposing engine: {str(e)}")

def load_table_info_from_file(file_path: Optional[str] = None) -> str:
    """
    Load table information from a previously saved file.
    
    Args:
        file_path (str, optional): Path to the table info file.
                                 Defaults to 'table_info_new.txt' in current directory
    
    Returns:
        str: The table information as a string
        
    Raises:
        FileNotFoundError: If the table info file doesn't exist
    """
    if file_path is None:
        file_path = Path(__file__).parent / "table_info_new.txt"
    else:
        file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Table info file not found: {file_path}")
    
    logger.info(f"Loading table info from: {file_path}")
    return file_path.read_text(encoding='utf-8')

# Example usage and testing
if __name__ == "__main__":
    # Example database URLs (replace with your actual database URL)
    
    # PostgreSQL example
    # DATABASE_URL = "postgresql://username:password@localhost:5432/database_name"
    
    # SQLite example
    # DATABASE_URL = "sqlite:///./test.db"
    
    # MySQL example
    # DATABASE_URL = "mysql+pymysql://username:password@localhost:3306/database_name"
    
    # Example usage with comprehensive error handling:
    try:
        DATABASE_URL='mssql+pyodbc://sa:Esap.12.Three@176.9.16.194:1433/JustForRestore?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=no'
        
        # Generate table info with default settings
        table_info = generate_table_info(DATABASE_URL)
        print("✅ Table info generated successfully!")
        print(f"First 500 characters:\n{table_info[:500]}...")
        
    except ValueError as e:
        print(f"❌ Error: {e}")