import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Body, APIRouter
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic Models for request/response validation
class DBConfig(BaseModel):
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    schema: str = "public"

# Legacy models for backward compatibility (will be removed after migration)
class UserConfigRequestLegacy(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Unique user identifier")
    db_config: DBConfig
    access_level: int = Field(..., ge=0, le=2, description="Access level: 0, 1, or 2")
    accessible_tables: Optional[List[str]] = Field(default=[], description="List of accessible table names")
    table_names: Optional[List[str]] = Field(default=[], description="List of table names for this configuration")

class UserConfigUpdateRequestLegacy(BaseModel):
    db_config: Optional[DBConfig] = Field(default=None, description="Database connection configuration")
    access_level: Optional[int] = Field(default=None, ge=0, le=2, description="Access level: 0, 1, or 2")
    accessible_tables: Optional[List[str]] = Field(default=None, description="List of accessible table names")
    table_names: Optional[List[str]] = Field(default=None, description="List of table names for this configuration")

class UserConfigResponseLegacy(BaseModel):
    config_id: Optional[int] = None
    user_id: str
    db_config: Dict[str, Any]
    access_level: int
    accessible_tables: List[str]
    table_names: List[str]
    is_latest: Optional[bool] = None
    created_at: Optional[str]
    updated_at: Optional[str]

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None

# New Pydantic Models for database configurations
class DatabaseConfigRequest(BaseModel):
    db_config: DBConfig = Field(..., description="Database connection configuration")

class DatabaseConfigUpdateRequest(BaseModel):
    db_config: DBConfig = Field(..., description="Database connection configuration")

class DatabaseConfigResponse(BaseModel):
    db_id: int
    db_config: Dict[str, Any]
    created_at: Optional[str]
    updated_at: Optional[str]

# Updated User Config Models to use db_id
class UserConfigRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Unique user identifier")
    db_id: int = Field(..., description="Database configuration ID from database_configs table")
    access_level: int = Field(..., ge=0, le=2, description="Access level: 0, 1, or 2")
    accessible_tables: Optional[List[str]] = Field(default=[], description="List of accessible table names")
    table_names: Optional[List[str]] = Field(default=[], description="List of table names for this configuration")

class UserConfigUpdateRequest(BaseModel):
    db_id: Optional[int] = Field(default=None, description="Database configuration ID from database_configs table")
    access_level: Optional[int] = Field(default=None, ge=0, le=2, description="Access level: 0, 1, or 2")
    accessible_tables: Optional[List[str]] = Field(default=None, description="List of accessible table names")
    table_names: Optional[List[str]] = Field(default=None, description="List of table names for this configuration")

class UserConfigResponse(BaseModel):
    config_id: Optional[int] = None
    user_id: str
    db_id: int
    db_config: Dict[str, Any]
    access_level: int
    accessible_tables: List[str]
    table_names: List[str]
    is_latest: Optional[bool] = None
    created_at: Optional[str]
    updated_at: Optional[str]

class UserConfigsResponse(BaseModel):
    user_id: str
    configs: List[UserConfigResponse]
    total_count: int
    latest_config_id: Optional[int] = None

class DatabaseManager:
    def __init__(self, host: str, port: int, user: str, password: str, default_db: str = 'postgres'):
        """
        Initialize database manager with connection parameters
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.default_db = default_db
        self.target_db = 'main_db'
        
    def get_connection(self, database: str = None):
        """
        Create database connection
        """
        db_name = database or self.target_db
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=db_name,
                user=self.user,
                password=self.password
            )
            return conn
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database {db_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database connection failed: {str(e)}"
            )
    
    def database_exists(self, db_name: str) -> bool:
        """
        Check if database exists
        """
        try:
            conn = self.get_connection(self.default_db)
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                    (db_name,)
                )
                exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except psycopg2.Error as e:
            logger.error(f"Error checking database existence: {e}")
            return False
    
    def create_database(self, db_name: str):
        """
        Create database if it doesn't exist
        """
        if self.database_exists(db_name):
            logger.info(f"Database '{db_name}' already exists")
            return True
        
        try:
            # Connect to default database to create new one
            conn = self.get_connection(self.default_db)
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database '{db_name}' created successfully")
            
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating database '{db_name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create database: {str(e)}"
            )
    
    def check_knowledge_base_tables_exist(self, db_name: str, db_config: DBConfig) -> Dict[str, bool]:
        """
        Check if all required knowledge base tables exist in the database
        """
        try:
            # Connect to the specific database
            conn = psycopg2.connect(
                host=db_config.DB_HOST,
                port=db_config.DB_PORT,
                database=db_name,
                user=db_config.DB_USER,
                password=db_config.DB_PASSWORD
            )
            
            with conn.cursor() as cursor:
                # Check if each table exists
                tables_to_check = [
                    'document_files',
                    'document_chunks', 
                    'intent_chunks',
                    'sub_intents'
                ]
                
                table_status = {}
                for table_name in tables_to_check:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = %s
                        );
                    """, (table_name,))
                    exists = cursor.fetchone()[0]
                    table_status[table_name] = exists
                
                # Check if vector extension exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_extension 
                        WHERE extname = 'vector'
                    );
                """)
                vector_exists = cursor.fetchone()[0]
                table_status['vector_extension'] = vector_exists
                
            conn.close()
            return table_status
            
        except psycopg2.Error as e:
            logger.error(f"Error checking knowledge base tables in database '{db_name}': {e}")
            return {}

    def upgrade_existing_tables(self, db_name: str, db_config: DBConfig):
        """
        Upgrade existing tables to ensure they have all required columns
        """
        try:
            # Connect to the specific database
            conn = psycopg2.connect(
                host=db_config.DB_HOST,
                port=db_config.DB_PORT,
                database=db_name,
                user=db_config.DB_USER,
                password=db_config.DB_PASSWORD
            )
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                logger.info(f"Checking and upgrading existing tables in database '{db_name}'")
                
                # Check and add missing columns to document_files table
                document_files_columns = [
                    ('file_description', 'TEXT'),
                    ('table_name', 'VARCHAR(200)'),
                    ('user_id', 'VARCHAR(100)'),
                    ('sub_intent_id', 'INTEGER')
                ]
                
                for column_name, column_type in document_files_columns:
                    try:
                        cursor.execute(f"""
                            ALTER TABLE document_files 
                            ADD COLUMN {column_name} {column_type}
                        """)
                        logger.info(f"✅ Added column '{column_name}' to document_files table")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"ℹ️ Column '{column_name}' already exists in document_files table")
                        else:
                            logger.warning(f"⚠️ Could not add column '{column_name}' to document_files table: {e}")
                
                # Check and add missing columns to document_chunks table
                document_chunks_columns = [
                    ('intent_id', 'VARCHAR(50)'),
                    ('mapped_to_intent', 'BOOLEAN DEFAULT FALSE')
                ]
                
                for column_name, column_type in document_chunks_columns:
                    try:
                        cursor.execute(f"""
                            ALTER TABLE document_chunks 
                            ADD COLUMN {column_name} {column_type}
                        """)
                        logger.info(f"✅ Added column '{column_name}' to document_chunks table")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info(f"ℹ️ Column '{column_name}' already exists in document_chunks table")
                        else:
                            logger.warning(f"⚠️ Could not add column '{column_name}' to document_chunks table: {e}")
                
                logger.info(f"✅ Table upgrade completed for database '{db_name}'")
            
            conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Error upgrading tables in database '{db_name}': {e}")
            return False

    def create_knowledge_base_tables(self, db_name: str, db_config: DBConfig):
        """
        Create all required tables and indexes for the knowledge base system
        """
        try:
            # Connect to the specific database
            conn = psycopg2.connect(
                host=db_config.DB_HOST,
                port=db_config.DB_PORT,
                database=db_name,
                user=db_config.DB_USER,
                password=db_config.DB_PASSWORD
            )
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                logger.info(f"Creating knowledge base tables in database '{db_name}'")
                
                # 1. Create vector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("✅ Vector extension created/verified")
                
                # 2. Create document_files table
                create_files_table_sql = """
                CREATE TABLE IF NOT EXISTS document_files (
                    file_id VARCHAR(255) PRIMARY KEY,
                    file_name VARCHAR(500) NOT NULL,
                    file_type VARCHAR(50),
                    file_path TEXT,
                    extracted_text TEXT,
                    full_summary TEXT,
                    title TEXT,
                    keywords TEXT[],
                    date_range JSONB,
                    processing_timestamp TIMESTAMP,
                    intent VARCHAR(255),
                    sub_intent VARCHAR(255),
                    title_summary_combined TEXT,
                    title_summary_embedding vector(1536),
                    file_description TEXT,
                    table_name VARCHAR(200),
                    user_id VARCHAR(100),
                    sub_intent_id INTEGER
                );
                """
                cursor.execute(create_files_table_sql)
                logger.info("✅ document_files table created")
                
                # 3. Create document_chunks table
                create_chunks_table_sql = """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id VARCHAR(255) PRIMARY KEY,
                    file_id VARCHAR(255) REFERENCES document_files(file_id) ON DELETE CASCADE,
                    chunk_text TEXT,
                    summary TEXT,
                    title TEXT,
                    keywords TEXT[],
                    date_range JSONB,
                    chunk_order INTEGER,
                    embedding vector(1536),
                    combined_context TEXT,
                    combined_embedding vector(1536),
                    metadata JSONB,
                    intent_id VARCHAR(50),
                    mapped_to_intent BOOLEAN DEFAULT FALSE
                );
                """
                cursor.execute(create_chunks_table_sql)
                logger.info("✅ document_chunks table created")
                
                # 4. Create intent_chunks table
                create_intent_chunks_table_sql = """
                CREATE TABLE IF NOT EXISTS intent_chunks (
                    intent_id VARCHAR(50) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
                """
                cursor.execute(create_intent_chunks_table_sql)
                logger.info("✅ intent_chunks table created")
                
                # 5. Create sub_intents table
                create_sub_intents_table_sql = """
                CREATE TABLE IF NOT EXISTS sub_intents (
                    sub_intent_id SERIAL PRIMARY KEY,
                    title VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    file_ids TEXT[] DEFAULT '{}',
                    embedding vector(1536) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_sub_intents_table_sql)
                logger.info("✅ sub_intents table created")
                
                # 6. Add foreign key constraints (with proper error handling)
                try:
                    cursor.execute("""
                        ALTER TABLE document_chunks 
                        ADD CONSTRAINT fk_document_chunks_intent 
                        FOREIGN KEY (intent_id) REFERENCES intent_chunks(intent_id)
                    """)
                    logger.info("✅ Added foreign key constraint: fk_document_chunks_intent")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info("ℹ️ Foreign key constraint fk_document_chunks_intent already exists")
                    else:
                        logger.warning(f"⚠️ Could not add foreign key constraint fk_document_chunks_intent: {e}")
                
                try:
                    cursor.execute("""
                        ALTER TABLE document_files 
                        ADD CONSTRAINT fk_document_files_sub_intent 
                        FOREIGN KEY (sub_intent_id) REFERENCES sub_intents(sub_intent_id)
                    """)
                    logger.info("✅ Added foreign key constraint: fk_document_files_sub_intent")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info("ℹ️ Foreign key constraint fk_document_files_sub_intent already exists")
                    else:
                        logger.warning(f"⚠️ Could not add foreign key constraint fk_document_files_sub_intent: {e}")
                
                logger.info("✅ Foreign key constraints processed")
                
                # 7. Create basic indexes
                basic_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_files_file_id ON document_files(file_id);",
                    "CREATE INDEX IF NOT EXISTS idx_files_keywords ON document_files USING GIN(keywords);",
                    "CREATE INDEX IF NOT EXISTS idx_files_date_range ON document_files USING GIN(date_range);",
                    "CREATE INDEX IF NOT EXISTS idx_files_processing_timestamp ON document_files(processing_timestamp);",
                    "CREATE INDEX IF NOT EXISTS idx_files_user_id ON document_files(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON document_chunks(file_id);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON document_chunks USING GIN(keywords);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_date_range ON document_chunks USING GIN(date_range);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON document_chunks USING GIN(metadata);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_order ON document_chunks(chunk_order);",
                    "CREATE INDEX IF NOT EXISTS idx_intent_title ON intent_chunks(title);",
                    "CREATE INDEX IF NOT EXISTS idx_sub_intent_title ON sub_intents(title);"
                ]
                
                for index_sql in basic_indexes:
                    cursor.execute(index_sql)
                logger.info("✅ Basic indexes created")
                
                # 8. Create vector indexes (these might fail if no data exists yet)
                vector_indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_files_title_summary_embedding ON document_files USING hnsw (title_summary_embedding vector_cosine_ops);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_combined_embedding ON document_chunks USING hnsw (combined_embedding vector_cosine_ops);",
                    "CREATE INDEX IF NOT EXISTS idx_intent_embedding ON intent_chunks USING hnsw (embedding vector_cosine_ops);",
                    "CREATE INDEX IF NOT EXISTS idx_sub_intent_embedding ON sub_intents USING hnsw (embedding vector_cosine_ops);"
                ]
                
                for index_sql in vector_indexes:
                    try:
                        cursor.execute(index_sql)
                        logger.info(f"✅ Created vector index: {index_sql.split('idx_')[1].split(' ')[0]}")
                    except Exception as e:
                        logger.warning(f"⚠️ Vector index creation skipped (will create after data insertion): {e}")
                
                logger.info(f"✅ All knowledge base tables and indexes created successfully in database '{db_name}'")
                
                # Upgrade existing tables to ensure all columns exist
                self.upgrade_existing_tables(db_name, db_config)
            
            conn.close()
            return True
            
        except psycopg2.Error as e:
            logger.error(f"Error creating knowledge base tables in database '{db_name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create knowledge base tables: {str(e)}"
            )
    
    def create_user_config_table(self):
        """
        Create user_config table with required columns to support multiple configs per user
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'user_config'
                    );
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    # Create new table
                    create_table_query = """
                    CREATE TABLE user_config (
                        config_id SERIAL PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        db_id INTEGER,
                        db_config JSONB NOT NULL,
                        access_level INTEGER NOT NULL CHECK (access_level IN (0, 1, 2)),
                        accessible_tables TEXT[] DEFAULT ARRAY[]::TEXT[],
                        table_names TEXT[] DEFAULT ARRAY[]::TEXT[],
                        is_latest BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                    cursor.execute(create_table_query)
                    logger.info("✅ Created new user_config table")
                else:
                    # Table exists, check if db_id column exists
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'user_config' AND column_name = 'db_id'
                    """)
                    has_db_id = cursor.fetchone() is not None
                    
                    if not has_db_id:
                        # Add db_id column to existing table
                        cursor.execute("""
                            ALTER TABLE user_config 
                            ADD COLUMN db_id INTEGER;
                        """)
                        logger.info("✅ Added db_id column to existing user_config table")
                    else:
                        logger.info("ℹ️ user_config table already has db_id column")
                    
                    # Check if table_names column exists
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'user_config' AND column_name = 'table_names'
                    """)
                    has_table_names = cursor.fetchone() is not None
                    
                    if not has_table_names:
                        # Add table_names column to existing table
                        cursor.execute("""
                            ALTER TABLE user_config 
                            ADD COLUMN table_names TEXT[] DEFAULT ARRAY[]::TEXT[];
                        """)
                        logger.info("✅ Added table_names column to existing user_config table")
                    else:
                        logger.info("ℹ️ user_config table already has table_names column")
                
                # Create indexes (these will be created if they don't exist)
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_user_config_user_id ON user_config(user_id);",
                    "CREATE INDEX IF NOT EXISTS idx_user_config_db_id ON user_config(db_id);",
                    "CREATE INDEX IF NOT EXISTS idx_user_config_access_level ON user_config(access_level);",
                    "CREATE INDEX IF NOT EXISTS idx_user_config_created_at ON user_config(created_at);",
                    "CREATE INDEX IF NOT EXISTS idx_user_config_is_latest ON user_config(is_latest);",
                    "CREATE INDEX IF NOT EXISTS idx_user_config_user_latest ON user_config(user_id, is_latest);"
                ]
                
                for index_sql in indexes:
                    try:
                        cursor.execute(index_sql)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not create index: {e}")
                
                conn.commit()
                logger.info("Table 'user_config' setup completed successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create table: {str(e)}"
            )

    def create_database_ownership_table(self):
        """
        Create database_ownership table to track who owns which databases
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS database_ownership (
            id SERIAL PRIMARY KEY,
            db_name VARCHAR(255) UNIQUE NOT NULL,
            owner_user_id VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_database_ownership_db_name ON database_ownership(db_name);
        CREATE INDEX IF NOT EXISTS idx_database_ownership_owner ON database_ownership(owner_user_id);
        CREATE INDEX IF NOT EXISTS idx_database_ownership_active ON database_ownership(is_active);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'database_ownership' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating database ownership table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create database ownership table: {str(e)}"
            )

    def create_database_configs_table(self):
        """
        Create database_configs table to store database configurations
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS database_configs (
            db_id SERIAL PRIMARY KEY,
            db_config JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_database_configs_created_at ON database_configs(created_at);
        CREATE INDEX IF NOT EXISTS idx_database_configs_updated_at ON database_configs(updated_at);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'database_configs' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating database configs table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create database configs table: {str(e)}"
            )

    def add_foreign_key_constraints(self):
        """
        Add foreign key constraints after all tables are created
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if foreign key constraint already exists
                cursor.execute("""
                    SELECT constraint_name 
                    FROM information_schema.table_constraints 
                    WHERE table_name = 'user_config' 
                    AND constraint_name = 'fk_user_config_db_id'
                """)
                constraint_exists = cursor.fetchone() is not None
                
                if not constraint_exists:
                    try:
                        cursor.execute("""
                            ALTER TABLE user_config 
                            ADD CONSTRAINT fk_user_config_db_id 
                            FOREIGN KEY (db_id) REFERENCES database_configs(db_id) ON DELETE SET NULL;
                        """)
                        conn.commit()
                        logger.info("✅ Foreign key constraint fk_user_config_db_id added successfully")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            logger.info("ℹ️ Foreign key constraint already exists")
                        else:
                            logger.warning(f"⚠️ Could not add foreign key constraint: {e}")
                else:
                    logger.info("ℹ️ Foreign key constraint fk_user_config_db_id already exists")
            
            conn.close()
        except Exception as e:
            logger.warning(f"Could not add foreign key constraints: {e}")
            # Don't raise exception, just log the warning
    
    def setup_database(self):
        """
        Setup complete database structure
        """
        # Create database
        self.create_database(self.target_db)
        
        # Create tables in correct order (database_configs first, then user_config)
        self.create_database_configs_table()
        self.create_database_ownership_table()
        self.create_user_config_table()
        
        # Try to add foreign key constraint after all tables are created
        self.add_foreign_key_constraints()
        
        # Migrate existing data if needed
        self.migrate_existing_data()
        
        logger.info("Database setup completed successfully")
        return True

    def get_database_config_by_id(self, db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get database configuration by db_id
        """
        select_query = """
        SELECT db_id, db_config, created_at, updated_at
        FROM database_configs
        WHERE db_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (db_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                return config
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting database config by id: {e}")
            return None

    def migrate_existing_data(self):
        """
        Migrate existing data from old structure to new structure
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if we need to migrate (if config_id column doesn't exist)
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_config' AND column_name = 'config_id'
                """)
                has_config_id = cursor.fetchone() is not None
                
                if not has_config_id:
                    logger.info("Migrating existing user_config table to new structure...")
                    
                    # Create new table with new structure
                    cursor.execute("""
                        CREATE TABLE user_config_new (
                            config_id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            db_id INTEGER,
                            db_config JSONB NOT NULL,
                            access_level INTEGER NOT NULL CHECK (access_level IN (0, 1, 2)),
                            accessible_tables TEXT[] DEFAULT ARRAY[]::TEXT[],
                            table_names TEXT[] DEFAULT ARRAY[]::TEXT[],
                            is_latest BOOLEAN DEFAULT FALSE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    
                    # Copy existing data and mark as latest
                    cursor.execute("""
                        INSERT INTO user_config_new (user_id, db_config, access_level, accessible_tables, is_latest, created_at, updated_at)
                        SELECT user_id, db_config, access_level, accessible_tables, TRUE, created_at, updated_at
                        FROM user_config;
                    """)
                    
                    # Drop old table and rename new one
                    cursor.execute("DROP TABLE user_config;")
                    cursor.execute("ALTER TABLE user_config_new RENAME TO user_config;")
                    
                    # Create indexes
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_user_id ON user_config(user_id);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_db_id ON user_config(db_id);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_access_level ON user_config(access_level);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_created_at ON user_config(created_at);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_is_latest ON user_config(is_latest);")
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_user_latest ON user_config(user_id, is_latest);")
                    
                    conn.commit()
                    logger.info("✅ Migration completed successfully")
                else:
                    logger.info("ℹ️ Database already has new structure, no migration needed")
                
                # Check if table_names column exists, if not add it
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_config' AND column_name = 'table_names'
                """)
                has_table_names = cursor.fetchone() is not None
                
                if not has_table_names:
                    logger.info("Adding table_names column to existing user_config table...")
                    cursor.execute("""
                        ALTER TABLE user_config 
                        ADD COLUMN table_names TEXT[] DEFAULT ARRAY[]::TEXT[];
                    """)
                    conn.commit()
                    logger.info("✅ table_names column added successfully")
                else:
                    logger.info("ℹ️ table_names column already exists")
                
                # Check if db_id column exists, if not add it
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_config' AND column_name = 'db_id'
                """)
                has_db_id = cursor.fetchone() is not None
                
                if not has_db_id:
                    logger.info("Adding db_id column to existing user_config table...")
                    cursor.execute("""
                        ALTER TABLE user_config 
                        ADD COLUMN db_id INTEGER;
                    """)
                    
                    # Create index for db_id
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_config_db_id ON user_config(db_id);")
                    
                    conn.commit()
                    logger.info("✅ db_id column added successfully")
                else:
                    logger.info("ℹ️ db_id column already exists")
                    
            conn.close()
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            # Don't raise exception, just log the error
            # The system can still work with the new structure
    
    def check_config_exists(self, user_config: UserConfigRequest) -> Optional[int]:
        """
        Check if a configuration already exists for a user
        Returns config_id if found, None if not found
        """
        check_query = """
        SELECT config_id 
        FROM user_config 
        WHERE user_id = %s 
        AND db_id = %s 
        AND access_level = %s 
        AND accessible_tables = %s
        AND table_names = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(check_query, (
                    user_config.user_id,
                    user_config.db_id,
                    user_config.access_level,
                    user_config.accessible_tables,
                    user_config.table_names or []
                ))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            return None
        except psycopg2.Error as e:
            logger.error(f"Error checking if config exists: {e}")
            return None

    def insert_user_config(self, user_config: UserConfigRequest) -> int:
        """
        Insert new user configuration and mark it as latest
        If configuration already exists, just mark it as latest
        Returns the config_id of the configuration (new or existing)
        """
        # First validate that the db_id exists
        db_config_data = self.get_database_config_by_id(user_config.db_id)
        if not db_config_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {user_config.db_id} not found"
            )
        
        # First check if this configuration already exists
        existing_config_id = self.check_config_exists(user_config)
        
        if existing_config_id:
            # Configuration already exists, just mark it as latest
            update_latest_query = """
            UPDATE user_config 
            SET is_latest = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s;
            """
            
            mark_existing_latest_query = """
            UPDATE user_config 
            SET is_latest = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            try:
                conn = self.get_connection()
                with conn.cursor() as cursor:
                    # First set all existing configs for this user as not latest
                    cursor.execute(update_latest_query, (user_config.user_id,))
                    
                    # Then mark the existing config as latest
                    cursor.execute(mark_existing_latest_query, (existing_config_id,))
                    conn.commit()
                    
                conn.close()
                logger.info(f"Existing user config {existing_config_id} for '{user_config.user_id}' marked as latest")
                return existing_config_id
            except psycopg2.Error as e:
                logger.error(f"Error updating existing user config: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update user configuration: {str(e)}"
                )
        
        # Configuration doesn't exist, create new one
        # First, set all existing configs for this user as not latest
        update_latest_query = """
        UPDATE user_config 
        SET is_latest = FALSE, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s;
        """
        
        # Then insert the new configuration as latest
        insert_query = """
        INSERT INTO user_config (user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
        RETURNING config_id;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # First update existing configs to not latest
                cursor.execute(update_latest_query, (user_config.user_id,))
                
                # Then insert new config as latest
                cursor.execute(insert_query, (
                    user_config.user_id,
                    user_config.db_id,
                    json.dumps(db_config_data['db_config']), 
                    user_config.access_level, 
                    user_config.accessible_tables,
                    user_config.table_names or []
                ))
                
                # Get the new config_id
                new_config_id = cursor.fetchone()[0]
                conn.commit()
                
            conn.close()
            logger.info(f"New user config created for '{user_config.user_id}' with config_id: {new_config_id}")
            return new_config_id
        except psycopg2.Error as e:
            logger.error(f"Error inserting user config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save user configuration: {str(e)}"
            )
    
    def get_user_config(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest user configuration by user_id
        """
        select_query = """
        SELECT config_id, user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, created_at, updated_at
        FROM user_config
        WHERE user_id = %s AND is_latest = TRUE
        ORDER BY created_at DESC
        LIMIT 1;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                # Convert to regular dict and handle datetime serialization
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                return config
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting user config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user configuration: {str(e)}"
            )
    
    def get_all_user_configs(self) -> List[Dict[str, Any]]:
        """
        Get latest configuration for all users
        """
        select_query = """
        SELECT config_id, user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, created_at, updated_at
        FROM user_config
        WHERE is_latest = TRUE
        ORDER BY created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query)
                results = cursor.fetchall()
            conn.close()
            
            configs = []
            for result in results:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                configs.append(config)
            
            return configs
        except psycopg2.Error as e:
            logger.error(f"Error getting all user configs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user configurations: {str(e)}"
            )

    def get_all_configs_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all configurations for a specific user
        """
        select_query = """
        SELECT config_id, user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, created_at, updated_at
        FROM user_config
        WHERE user_id = %s
        ORDER BY created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id,))
                results = cursor.fetchall()
            conn.close()
            
            configs = []
            for result in results:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                configs.append(config)
            
            return configs
        except psycopg2.Error as e:
            logger.error(f"Error getting all configs for user: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user configurations: {str(e)}"
            )

    def get_config_by_id(self, config_id: int) -> Optional[Dict[str, Any]]:
        """
        Get specific configuration by config_id
        """
        select_query = """
        SELECT config_id, user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, created_at, updated_at
        FROM user_config
        WHERE config_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (config_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                return config
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting config by id: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve configuration: {str(e)}"
            )



    def delete_config_by_id(self, config_id: int) -> bool:
        """
        Delete specific configuration by config_id
        If the deleted config was the latest, make the most recent remaining config the latest
        """
        # First get the config to check if it's the latest and get user_id
        config = self.get_config_by_id(config_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with config_id {config_id} not found"
            )
        
        user_id = config['user_id']
        was_latest = config['is_latest']
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Delete the configuration
                cursor.execute("DELETE FROM user_config WHERE config_id = %s", (config_id,))
                
                # If it was the latest config, make the most recent remaining config the latest
                if was_latest:
                    update_latest_query = """
                    UPDATE user_config 
                    SET is_latest = TRUE, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s 
                    AND config_id = (
                        SELECT config_id 
                        FROM user_config 
                        WHERE user_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    );
                    """
                    cursor.execute(update_latest_query, (user_id, user_id))
                
                conn.commit()
            conn.close()
            
            logger.info(f"Configuration {config_id} for user '{user_id}' deleted successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete configuration: {str(e)}"
            )

    def delete_database(self, db_name: str):
        """
        Delete database if it exists
        """
        if not self.database_exists(db_name):
            logger.info(f"Database '{db_name}' does not exist")
            return True
        
        try:
            # Connect to default database to drop the target database
            conn = self.get_connection(self.default_db)
            conn.autocommit = True
            
            with conn.cursor() as cursor:
                # Terminate all connections to the database first
                cursor.execute(f"""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                """, (db_name,))
                
                # Drop the database
                cursor.execute(f'DROP DATABASE "{db_name}"')
                logger.info(f"Database '{db_name}' deleted successfully")
            
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting database '{db_name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete database: {str(e)}"
            )

    def check_and_create_user_database(self, user_config: UserConfigRequest) -> bool:
        """
        Check if user's database exists and create it if not
        """
        # Get the database configuration from db_id
        db_config_data = self.get_database_config_by_id(user_config.db_id)
        if not db_config_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {user_config.db_id} not found"
            )
        
        db_config = db_config_data['db_config']
        db_name = db_config['DB_NAME']
        
        if not self.database_exists(db_name):
            logger.info(f"User database '{db_name}' does not exist, creating it...")
            success = self.create_database(db_name)
            if success:
                # Record ownership
                self.record_database_ownership(db_name, user_config.user_id)
                # Create knowledge base tables for the new database
                self.create_knowledge_base_tables(db_name, DBConfig(**db_config))
            return success
        else:
            logger.info(f"User database '{db_name}' already exists")
            # Even if database exists, ensure tables exist and are upgraded
            try:
                self.create_knowledge_base_tables(db_name, DBConfig(**db_config))
                logger.info(f"Knowledge base tables ensured in existing database '{db_name}'")
            except Exception as e:
                logger.warning(f"Could not ensure tables in existing database '{db_name}': {e}")
                # Try to upgrade existing tables even if creation failed
                try:
                    self.upgrade_existing_tables(db_name, DBConfig(**db_config))
                    logger.info(f"Table upgrade completed for existing database '{db_name}'")
                except Exception as upgrade_e:
                    logger.warning(f"Could not upgrade tables in existing database '{db_name}': {upgrade_e}")
            return True

    def record_database_ownership(self, db_name: str, owner_user_id: str) -> bool:
        """
        Record database ownership in the tracking table
        """
        insert_query = """
        INSERT INTO database_ownership (db_name, owner_user_id)
        VALUES (%s, %s)
        ON CONFLICT (db_name) 
        DO UPDATE SET 
            owner_user_id = EXCLUDED.owner_user_id,
            is_active = TRUE;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(insert_query, (db_name, owner_user_id))
                conn.commit()
            conn.close()
            logger.info(f"Database ownership recorded: '{db_name}' -> '{owner_user_id}'")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error recording database ownership: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to record database ownership: {str(e)}"
            )

    def get_database_owner(self, db_name: str) -> Optional[str]:
        """
        Get the owner of a database
        """
        select_query = """
        SELECT owner_user_id FROM database_ownership 
        WHERE db_name = %s AND is_active = TRUE;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(select_query, (db_name,))
                result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else None
        except psycopg2.Error as e:
            logger.error(f"Error getting database owner: {e}")
            return None

    def get_user_databases(self, user_id: str) -> List[str]:
        """
        Get all databases owned by a user
        """
        select_query = """
        SELECT db_name FROM database_ownership 
        WHERE owner_user_id = %s AND is_active = TRUE
        ORDER BY created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(select_query, (user_id,))
                results = cursor.fetchall()
            conn.close()
            
            return [result[0] for result in results]
        except psycopg2.Error as e:
            logger.error(f"Error getting user databases: {e}")
            return []

    def mark_database_deleted(self, db_name: str) -> bool:
        """
        Mark database as deleted in ownership table
        """
        update_query = """
        UPDATE database_ownership 
        SET is_active = FALSE 
        WHERE db_name = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (db_name,))
                conn.commit()
            conn.close()
            logger.info(f"Database '{db_name}' marked as deleted in ownership table")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error marking database as deleted: {e}")
            return False

    def get_user_unique_tables(self, user_id: str) -> Union[List[str], bool]:
        """
        Get unique table names from user's database document_files table
        
        Args:
            user_id: The user ID to get configuration for
            
        Returns:
            List of unique table names if successful, False if any error occurs
        """
        try:
            # Step 1: Get user configuration
            user_config = self.get_user_config(user_id)
            if not user_config:
                logger.warning(f"User configuration not found for user_id: {user_id}")
                return False
            
            # Step 2: Extract database configuration
            db_config_data = user_config.get('db_config', {})
            if not db_config_data:
                logger.warning(f"No database configuration found for user_id: {user_id}")
                return False
            
            db_name = db_config_data.get('DB_NAME')
            if not db_name:
                logger.warning(f"No database name found in configuration for user_id: {user_id}")
                return False
            
            # Step 3: Check if database exists
            if not self.database_exists(db_name):
                logger.warning(f"Database '{db_name}' does not exist for user_id: {user_id}")
                return False
            
            # Step 4: Connect to user's database and query document_files table
            try:
                conn = psycopg2.connect(
                    host=db_config_data.get('DB_HOST'),
                    port=db_config_data.get('DB_PORT'),
                    database=db_name,
                    user=db_config_data.get('DB_USER'),
                    password=db_config_data.get('DB_PASSWORD')
                )
                
                with conn.cursor() as cursor:
                    # Check if document_files table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = 'document_files'
                        );
                    """)
                    table_exists = cursor.fetchone()[0]
                    
                    if not table_exists:
                        logger.warning(f"document_files table does not exist in database '{db_name}' for user_id: {user_id}")
                        conn.close()
                        return False
                    
                    # Query unique table_name values from document_files table
                    cursor.execute("""
                        SELECT DISTINCT table_name 
                        FROM document_files 
                        WHERE table_name IS NOT NULL 
                        AND table_name != ''
                        ORDER BY table_name;
                    """)
                    
                    results = cursor.fetchall()
                    unique_tables = [row[0] for row in results if row[0]]
                    
                    conn.close()
                    
                    logger.info(f"Found {len(unique_tables)} unique tables for user_id: {user_id} in database '{db_name}'")
                    return unique_tables
                    
            except psycopg2.Error as e:
                logger.error(f"Database connection error for user_id {user_id}: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Error getting unique tables for user_id {user_id}: {e}")
            return False

    def get_config_table_names(self, config_id: int) -> Optional[List[str]]:
        """
        Get table names for a specific configuration
        
        Args:
            config_id: The configuration ID to get table names for
            
        Returns:
            List of table names if found, None if config doesn't exist
        """
        try:
            config = self.get_config_by_id(config_id)
            if not config:
                logger.warning(f"Configuration not found for config_id: {config_id}")
                return None
            
            table_names = config.get('table_names', [])
            logger.info(f"Retrieved {len(table_names)} table names for config_id: {config_id}")
            return table_names
            
        except Exception as e:
            logger.error(f"Error getting table names for config_id {config_id}: {e}")
            return None

    def append_table_name(self, config_id: int, table_name: str) -> bool:
        """
        Append a table name to the table_names list for a specific configuration
        
        Args:
            config_id: The configuration ID to append table name to
            table_name: The table name to append
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if config exists
            config = self.get_config_by_id(config_id)
            if not config:
                logger.warning(f"Configuration not found for config_id: {config_id}")
                return False
            
            # Get current table names
            current_table_names = config.get('table_names', [])
            
            # Check if table name already exists
            if table_name in current_table_names:
                logger.info(f"Table name '{table_name}' already exists in config_id: {config_id}")
                return True
            
            # Append the new table name
            new_table_names = current_table_names + [table_name]
            
            # Update the database
            update_query = """
            UPDATE user_config 
            SET table_names = %s, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (new_table_names, config_id))
                conn.commit()
            conn.close()
            
            logger.info(f"Successfully appended table name '{table_name}' to config_id: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error appending table name '{table_name}' to config_id {config_id}: {e}")
            return False

    def delete_table_name(self, config_id: int, table_name: str) -> bool:
        """
        Delete a specific table name from the table_names list for a configuration
        
        Args:
            config_id: The configuration ID to delete table name from
            table_name: The table name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if config exists
            config = self.get_config_by_id(config_id)
            if not config:
                logger.warning(f"Configuration not found for config_id: {config_id}")
                return False
            
            # Get current table names
            current_table_names = config.get('table_names', [])
            
            # Check if table name exists
            if table_name not in current_table_names:
                logger.warning(f"Table name '{table_name}' not found in config_id: {config_id}")
                return False
            
            # Remove the table name
            new_table_names = [name for name in current_table_names if name != table_name]
            
            # Update the database
            update_query = """
            UPDATE user_config 
            SET table_names = %s, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (new_table_names, config_id))
                conn.commit()
            conn.close()
            
            logger.info(f"Successfully deleted table name '{table_name}' from config_id: {config_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting table name '{table_name}' from config_id {config_id}: {e}")
            return False

    def get_user_table_names(self, user_id: str) -> Optional[List[str]]:
        """
        Get table names for a user's current configuration (is_latest=True)
        
        Args:
            user_id: The user ID to get table names for
            
        Returns:
            List of table names if found, None if user doesn't exist
        """
        try:
            # Get the current config for the user
            config = self.get_user_config(user_id)
            if not config:
                logger.warning(f"User configuration not found for user_id: {user_id}")
                return None
            
            table_names = config.get('table_names', [])
            logger.info(f"Retrieved {len(table_names)} table names for user_id: {user_id}")
            return table_names
            
        except Exception as e:
            logger.error(f"Error getting table names for user_id {user_id}: {e}")
            return None

    def append_user_table_name(self, user_id: str, table_name: str) -> bool:
        """
        Append a table name to the table_names list for a user's current configuration
        
        Args:
            user_id: The user ID to append table name to
            table_name: The table name to append
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the current config for the user
            config = self.get_user_config(user_id)
            if not config:
                logger.warning(f"User configuration not found for user_id: {user_id}")
                return False
            
            config_id = config.get('config_id')
            if not config_id:
                logger.warning(f"No config_id found for user_id: {user_id}")
                return False
            
            # Get current table names
            current_table_names = config.get('table_names', [])
            
            # Check if table name already exists
            if table_name in current_table_names:
                logger.info(f"Table name '{table_name}' already exists for user_id: {user_id}")
                return True
            
            # Append the new table name
            new_table_names = current_table_names + [table_name]
            
            # Update the database
            update_query = """
            UPDATE user_config 
            SET table_names = %s, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (new_table_names, config_id))
                conn.commit()
            conn.close()
            
            logger.info(f"Successfully appended table name '{table_name}' for user_id: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error appending table name '{table_name}' for user_id {user_id}: {e}")
            return False

    def delete_user_table_name(self, user_id: str, table_name: str) -> bool:
        """
        Delete a specific table name from the table_names list for a user's current configuration
        
        Args:
            user_id: The user ID to delete table name from
            table_name: The table name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the current config for the user
            config = self.get_user_config(user_id)
            if not config:
                logger.warning(f"User configuration not found for user_id: {user_id}")
                return False
            
            config_id = config.get('config_id')
            if not config_id:
                logger.warning(f"No config_id found for user_id: {user_id}")
                return False
            
            # Get current table names
            current_table_names = config.get('table_names', [])
            
            # Check if table name exists
            if table_name not in current_table_names:
                logger.warning(f"Table name '{table_name}' not found for user_id: {user_id}")
                return False
            
            # Remove the table name
            new_table_names = [name for name in current_table_names if name != table_name]
            
            # Update the database
            update_query = """
            UPDATE user_config 
            SET table_names = %s, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (new_table_names, config_id))
                conn.commit()
            conn.close()
            
            logger.info(f"Successfully deleted table name '{table_name}' for user_id: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting table name '{table_name}' for user_id {user_id}: {e}")
            return False

    def get_all_table_names_by_db_id(self, db_id: int) -> Dict[str, Any]:
        """
        Get all table names from all user configurations that use a specific database configuration
        
        Args:
            db_id: The database configuration ID to get table names for
            
        Returns:
            Dictionary containing:
            - table_names: List of unique table names
            - configs_contributing: List of config_ids that contributed table names
            - total_configs: Number of configurations found for this db_id
            - None if db_id doesn't exist or no configurations found
        """
        try:
            # First validate that the db_id exists
            db_config_data = self.get_database_config_by_id(db_id)
            if not db_config_data:
                logger.warning(f"Database configuration with db_id {db_id} not found")
                return None
            
            # Query all configurations that use this db_id
            select_query = """
            SELECT config_id, user_id, table_names
            FROM user_config
            WHERE db_id = %s AND table_names IS NOT NULL AND array_length(table_names, 1) > 0
            ORDER BY config_id;
            """
            
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (db_id,))
                results = cursor.fetchall()
            conn.close()
            
            if not results:
                logger.info(f"No configurations with table names found for db_id: {db_id}")
                return {
                    "table_names": [],
                    "configs_contributing": [],
                    "total_configs": 0,
                    "db_id": db_id,
                    "unique_table_count": 0
                }
            
            # Collect all table names and track contributing configs
            all_table_names = set()  # Use set to avoid duplicates
            configs_contributing = []
            
            for result in results:
                config_id = result['config_id']
                table_names = result.get('table_names', [])
                
                if table_names:  # Only add if there are actual table names
                    all_table_names.update(table_names)
                    configs_contributing.append({
                        "config_id": config_id,
                        "user_id": result['user_id'],
                        "table_count": len(table_names)
                    })
            
            # Convert set back to sorted list
            unique_table_names = sorted(list(all_table_names))
            
            logger.info(f"Retrieved {len(unique_table_names)} unique table names from {len(configs_contributing)} configurations for db_id: {db_id}")
            
            return {
                "table_names": unique_table_names,
                "configs_contributing": configs_contributing,
                "total_configs": len(configs_contributing),
                "db_id": db_id,
                "unique_table_count": len(unique_table_names)
            }
            
        except Exception as e:
            logger.error(f"Error getting table names for db_id {db_id}: {e}")
            return None

    def set_user_config_by_config_id(self, user_id: str, config_id: int) -> Dict[str, Any]:
        """
        Set a user's current configuration to a specific config_id (make it the latest)
        
        Args:
            user_id: The user ID to set configuration for
            config_id: The configuration ID to set as current
            
        Returns:
            Dictionary containing operation results and metadata
        """
        try:
            # First get the configuration details by config_id
            config = self.get_config_by_id(config_id)
            if not config:
                logger.warning(f"Configuration with config_id {config_id} not found")
                return None
            
            # Verify the config belongs to the specified user
            if config['user_id'] != user_id:
                logger.warning(f"Configuration {config_id} does not belong to user {user_id}")
                return None
            
            # Create a UserConfigRequest object from the existing config
            user_config_request = UserConfigRequest(
                user_id=config['user_id'],
                db_id=config['db_id'],
                access_level=config['access_level'],
                accessible_tables=config.get('accessible_tables', []),
                table_names=config.get('table_names', [])
            )
            
            # Check if this configuration already exists (it should, since we got it by config_id)
            existing_config_id = self.check_config_exists(user_config_request)
            config_was_reused = existing_config_id is not None
            
            # Get database configuration details for additional processing
            db_config_data = self.get_database_config_by_id(config['db_id'])
            if not db_config_data:
                logger.warning(f"Database configuration with db_id {config['db_id']} not found")
                return None
            
            db_name = db_config_data['db_config']['DB_NAME']
            db_existed = self.database_exists(db_name)
            
            # Ensure user's database exists and has required tables
            self.check_and_create_user_database(user_config_request)
            
            # Mark this configuration as the latest for the user
            # First, set all existing configs for this user as not latest
            update_latest_query = """
            UPDATE user_config 
            SET is_latest = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s;
            """
            
            # Then mark the specified config as latest
            mark_config_latest_query = """
            UPDATE user_config 
            SET is_latest = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # First update existing configs to not latest
                cursor.execute(update_latest_query, (user_id,))
                
                # Then mark the specified config as latest
                cursor.execute(mark_config_latest_query, (config_id,))
                conn.commit()
            conn.close()
            
            # Check table status for better feedback
            table_status = self.check_knowledge_base_tables_exist(db_name, DBConfig(**db_config_data['db_config']))
            
            logger.info(f"Successfully set configuration {config_id} as latest for user '{user_id}'")
            
            return {
                "config_id": config_id,
                "user_id": user_id,
                "db_id": config['db_id'],
                "config_was_reused": config_was_reused,
                "database_created": not db_existed,
                "database_name": db_name,
                "table_status": table_status,
                "action": "set_as_latest"
            }
            
        except Exception as e:
            logger.error(f"Error setting user config by config_id: {e}")
            return None

    def get_configs_by_user_and_db(self, user_id: str, db_id: int) -> Dict[str, Any]:
        """
        Get all configurations for a specific user and database combination
        
        Args:
            user_id: The user ID to get configurations for
            db_id: The database configuration ID to filter by
            
        Returns:
            Dictionary containing:
            - configs: List of all configurations for this user-db combination
            - count: Number of configurations found
            - latest_config_id: ID of the latest configuration (if any)
            - None if no configurations found
        """
        try:
            # First validate that the db_id exists
            db_config_data = self.get_database_config_by_id(db_id)
            if not db_config_data:
                logger.warning(f"Database configuration with db_id {db_id} not found")
                return None
            
            # Query all configurations for this user and db_id combination
            select_query = """
            SELECT config_id, user_id, db_id, db_config, access_level, accessible_tables, table_names, is_latest, created_at, updated_at
            FROM user_config
            WHERE user_id = %s AND db_id = %s
            ORDER BY created_at DESC;
            """
            
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id, db_id))
                results = cursor.fetchall()
            conn.close()
            
            if not results:
                logger.info(f"No configurations found for user '{user_id}' with db_id {db_id}")
                return {
                    "configs": [],
                    "count": 0,
                    "latest_config_id": None,
                    "user_id": user_id,
                    "db_id": db_id,
                    "database_name": db_config_data['db_config']['DB_NAME']
                }
            
            # Process results and find the latest config
            configs = []
            latest_config_id = None
            
            for result in results:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                configs.append(config)
                
                # Track the latest config
                if config['is_latest']:
                    latest_config_id = config['config_id']
            
            logger.info(f"Retrieved {len(configs)} configurations for user '{user_id}' with db_id {db_id}")
            
            return {
                "configs": configs,
                "count": len(configs),
                "latest_config_id": latest_config_id,
                "user_id": user_id,
                "db_id": db_id,
                "database_name": db_config_data['db_config']['DB_NAME']
            }
            
        except Exception as e:
            logger.error(f"Error getting configs for user {user_id} with db_id {db_id}: {e}")
            return None

    def update_user_config(self, config_id: int, update_data: UserConfigUpdateRequest) -> bool:
        """
        Update user configuration fields for a specific config_id
        
        Args:
            config_id: The configuration ID to update
            update_data: The data to update (partial update supported)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if config exists
            existing_config = self.get_config_by_id(config_id)
            if not existing_config:
                logger.warning(f"Configuration not found for config_id: {config_id}")
                return False
            
            # Build dynamic update query based on provided fields
            update_fields = []
            update_values = []
            
            # Check each field and add to update if provided
            if update_data.db_id is not None:
                # Validate that the new db_id exists
                db_config_data = self.get_database_config_by_id(update_data.db_id)
                if not db_config_data:
                    logger.warning(f"Database configuration with db_id {update_data.db_id} not found")
                    return False
                
                update_fields.append("db_id = %s")
                update_values.append(update_data.db_id)
                update_fields.append("db_config = %s")
                update_values.append(json.dumps(db_config_data['db_config']))
                logger.info(f"Updating db_id to {update_data.db_id} for config_id: {config_id}")
            
            if update_data.access_level is not None:
                update_fields.append("access_level = %s")
                update_values.append(update_data.access_level)
                logger.info(f"Updating access_level to {update_data.access_level} for config_id: {config_id}")
            
            if update_data.accessible_tables is not None:
                update_fields.append("accessible_tables = %s")
                update_values.append(update_data.accessible_tables)
                logger.info(f"Updating accessible_tables for config_id: {config_id}")
            
            if update_data.table_names is not None:
                update_fields.append("table_names = %s")
                update_values.append(update_data.table_names)
                logger.info(f"Updating table_names for config_id: {config_id}")
            
            # If no fields to update, return early
            if not update_fields:
                logger.info(f"No fields to update for config_id: {config_id}")
                return True
            
            # Add updated_at timestamp
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            # Build the complete update query
            update_query = f"""
            UPDATE user_config 
            SET {', '.join(update_fields)}
            WHERE config_id = %s;
            """
            
            # Add config_id to values
            update_values.append(config_id)
            
            # Execute the update
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, update_values)
                conn.commit()
            conn.close()
            
            logger.info(f"Successfully updated configuration {config_id} with {len(update_fields)-1} fields")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration {config_id}: {e}")
            return False

    # Legacy support methods for backward compatibility
    def insert_user_config_legacy(self, user_config: UserConfigRequestLegacy) -> int:
        """
        Legacy method to insert user configuration with db_config (for backward compatibility)
        """
        # First check if this configuration already exists
        existing_config_id = self.check_config_exists_legacy(user_config)
        
        if existing_config_id:
            # Configuration already exists, just mark it as latest
            update_latest_query = """
            UPDATE user_config 
            SET is_latest = FALSE, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s;
            """
            
            mark_existing_latest_query = """
            UPDATE user_config 
            SET is_latest = TRUE, updated_at = CURRENT_TIMESTAMP
            WHERE config_id = %s;
            """
            
            try:
                conn = self.get_connection()
                with conn.cursor() as cursor:
                    # First set all existing configs for this user as not latest
                    cursor.execute(update_latest_query, (user_config.user_id,))
                    
                    # Then mark the existing config as latest
                    cursor.execute(mark_existing_latest_query, (existing_config_id,))
                    conn.commit()
                    
                conn.close()
                logger.info(f"Existing user config {existing_config_id} for '{user_config.user_id}' marked as latest")
                return existing_config_id
            except psycopg2.Error as e:
                logger.error(f"Error updating existing user config: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to update user configuration: {str(e)}"
                )
        
        # Configuration doesn't exist, create new one
        # First, set all existing configs for this user as not latest
        update_latest_query = """
        UPDATE user_config 
        SET is_latest = FALSE, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s;
        """
        
        # Then insert the new configuration as latest
        insert_query = """
        INSERT INTO user_config (user_id, db_config, access_level, accessible_tables, table_names, is_latest, updated_at)
        VALUES (%s, %s, %s, %s, %s, TRUE, CURRENT_TIMESTAMP)
        RETURNING config_id;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # First update existing configs to not latest
                cursor.execute(update_latest_query, (user_config.user_id,))
                
                # Then insert new config as latest
                cursor.execute(insert_query, (
                    user_config.user_id, 
                    json.dumps(user_config.db_config.dict()), 
                    user_config.access_level, 
                    user_config.accessible_tables,
                    user_config.table_names or []
                ))
                
                # Get the new config_id
                new_config_id = cursor.fetchone()[0]
                conn.commit()
                
            conn.close()
            logger.info(f"New user config created for '{user_config.user_id}' with config_id: {new_config_id}")
            return new_config_id
        except psycopg2.Error as e:
            logger.error(f"Error inserting user config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save user configuration: {str(e)}"
            )

    def check_config_exists_legacy(self, user_config: UserConfigRequestLegacy) -> Optional[int]:
        """
        Legacy method to check if a configuration already exists for a user
        Returns config_id if found, None if not found
        """
        check_query = """
        SELECT config_id 
        FROM user_config 
        WHERE user_id = %s 
        AND db_config = %s 
        AND access_level = %s 
        AND accessible_tables = %s
        AND table_names = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(check_query, (
                    user_config.user_id,
                    json.dumps(user_config.db_config.dict()),
                    user_config.access_level,
                    user_config.accessible_tables,
                    user_config.table_names or []
                ))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0]
            return None
        except psycopg2.Error as e:
            logger.error(f"Error checking if config exists: {e}")
            return None

    def check_and_create_user_database_legacy(self, user_config: UserConfigRequestLegacy) -> bool:
        """
        Legacy method to check if user's database exists and create it if not
        """
        db_name = user_config.db_config.DB_NAME
        
        if not self.database_exists(db_name):
            logger.info(f"User database '{db_name}' does not exist, creating it...")
            success = self.create_database(db_name)
            if success:
                # Record ownership
                self.record_database_ownership(db_name, user_config.user_id)
                # Create knowledge base tables for the new database
                self.create_knowledge_base_tables(db_name, user_config.db_config)
            return success
        else:
            logger.info(f"User database '{db_name}' already exists")
            # Even if database exists, ensure tables exist and are upgraded
            try:
                self.create_knowledge_base_tables(db_name, user_config.db_config)
                logger.info(f"Knowledge base tables ensured in existing database '{db_name}'")
            except Exception as e:
                logger.warning(f"Could not ensure tables in existing database '{db_name}': {e}")
                # Try to upgrade existing tables even if creation failed
                try:
                    self.upgrade_existing_tables(db_name, user_config.db_config)
                    logger.info(f"Table upgrade completed for existing database '{db_name}'")
                except Exception as upgrade_e:
                    logger.warning(f"Could not upgrade tables in existing database '{db_name}': {upgrade_e}")
            return True

    # Database Configs CRUD Methods
    def insert_database_config(self, db_config_request: DatabaseConfigRequest) -> int:
        """
        Insert new database configuration
        Returns the db_id of the new configuration
        """
        insert_query = """
        INSERT INTO database_configs (db_config, updated_at)
        VALUES (%s, CURRENT_TIMESTAMP)
        RETURNING db_id;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(insert_query, (json.dumps(db_config_request.db_config.dict()),))
                new_db_id = cursor.fetchone()[0]
                conn.commit()
            
            conn.close()
            logger.info(f"New database config created with db_id: {new_db_id}")
            return new_db_id
        except psycopg2.Error as e:
            logger.error(f"Error inserting database config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save database configuration: {str(e)}"
            )

    def get_database_config(self, db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get database configuration by db_id
        """
        select_query = """
        SELECT db_id, db_config, created_at, updated_at
        FROM database_configs
        WHERE db_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (db_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                return config
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting database config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve database configuration: {str(e)}"
            )

    def get_all_database_configs(self) -> List[Dict[str, Any]]:
        """
        Get all database configurations
        """
        select_query = """
        SELECT db_id, db_config, created_at, updated_at
        FROM database_configs
        ORDER BY created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query)
                results = cursor.fetchall()
            conn.close()
            
            configs = []
            for result in results:
                config = dict(result)
                config['created_at'] = config['created_at'].isoformat() if config['created_at'] else None
                config['updated_at'] = config['updated_at'].isoformat() if config['updated_at'] else None
                configs.append(config)
            
            return configs
        except psycopg2.Error as e:
            logger.error(f"Error getting all database configs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve database configurations: {str(e)}"
            )

    def update_database_config(self, db_id: int, update_data: DatabaseConfigUpdateRequest) -> bool:
        """
        Update database configuration by db_id
        """
        update_query = """
        UPDATE database_configs 
        SET db_config = %s, updated_at = CURRENT_TIMESTAMP
        WHERE db_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(update_query, (
                    json.dumps(update_data.db_config.dict()),
                    db_id
                ))
                rows_affected = cursor.rowcount
                conn.commit()
            conn.close()
            
            if rows_affected > 0:
                logger.info(f"Database config {db_id} updated successfully")
                return True
            else:
                logger.warning(f"Database config {db_id} not found for update")
                return False
        except psycopg2.Error as e:
            logger.error(f"Error updating database config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update database configuration: {str(e)}"
            )

    def delete_database_config(self, db_id: int) -> bool:
        """
        Delete database configuration by db_id
        """
        delete_query = """
        DELETE FROM database_configs 
        WHERE db_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (db_id,))
                rows_affected = cursor.rowcount
                conn.commit()
            conn.close()
            
            if rows_affected > 0:
                logger.info(f"Database config {db_id} deleted successfully")
                return True
            else:
                logger.warning(f"Database config {db_id} not found for deletion")
                return False
        except psycopg2.Error as e:
            logger.error(f"Error deleting database config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete database configuration: {str(e)}"
            )


# FastAPI App Setup
app = FastAPI(
    title="PostgreSQL Database Manager API",
    description="API for managing PostgreSQL database configurations with user access control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize database manager with your credentials
db_manager = DatabaseManager(
    host='localhost',
    port=5433,
    user='postgres',
    password='1234'
)

@router.on_event("startup")
async def startup_event():
    """
    Setup database on application startup
    """
    try:
        db_manager.setup_database()
        logger.info("Database setup completed on startup")
    except Exception as e:
        logger.warning(f"Could not setup database on startup: {e}")
        logger.info("You can setup the database by calling POST /setup")

@router.post("/setup", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def setup_database():
    """
    Setup database and table structure
    """
    try:
        db_manager.setup_database()
        return APIResponse(
            status="success",
            message="Database and table setup completed successfully"
        )
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setup failed: {str(e)}"
        )

@router.post("/setup/force", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def force_setup_database():
    """
    Force setup database and table structure (drops and recreates tables)
    """
    try:
        # Drop existing tables if they exist
        conn = db_manager.get_connection()
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS user_config CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS database_configs CASCADE;")
            cursor.execute("DROP TABLE IF EXISTS database_ownership CASCADE;")
            conn.commit()
        conn.close()
        
        # Setup fresh database
        db_manager.setup_database()
        return APIResponse(
            status="success",
            message="Database and table setup completed successfully (forced recreation)"
        )
    except Exception as e:
        logger.error(f"Force setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Force setup failed: {str(e)}"
        )

@router.post("/user-config", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def set_user_config(user_config: UserConfigRequest):
    """
    Create or update user configuration
    
    - **user_id**: Unique identifier for the user
    - **db_id**: Database configuration ID from database_configs table
    - **access_level**: Access level (0, 1, or 2)
    - **accessible_tables**: List of tables the user can access
    """
    try:
        # Get database configuration details from db_id
        db_config_data = db_manager.get_database_config_by_id(user_config.db_id)
        if not db_config_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {user_config.db_id} not found"
            )
        
        db_name = db_config_data['db_config']['DB_NAME']
        db_existed = db_manager.database_exists(db_name)
        
        # Check if configuration already exists before creating/updating
        existing_config_id = db_manager.check_config_exists(user_config)
        config_was_reused = existing_config_id is not None
        
        # First check if user's database exists and create it if not
        db_manager.check_and_create_user_database(user_config)
        
        # Then save the user configuration
        config_id = db_manager.insert_user_config(user_config)
        
        # Check table status for better feedback
        table_status = db_manager.check_knowledge_base_tables_exist(db_name, DBConfig(**db_config_data['db_config']))
        
        # Build appropriate message based on what happened
        if config_was_reused:
            if db_existed:
                message = f"User configuration for '{user_config.user_id}' reused existing config (ID: {config_id}). Database '{db_name}' already existed. Knowledge base tables ensured."
            else:
                message = f"User configuration for '{user_config.user_id}' reused existing config (ID: {config_id}). Database '{db_name}' created with knowledge base tables."
        else:
            if db_existed:
                message = f"User configuration for '{user_config.user_id}' created successfully (ID: {config_id}). Database '{db_name}' already existed. Knowledge base tables ensured."
            else:
                message = f"User configuration for '{user_config.user_id}' created successfully (ID: {config_id}). Database '{db_name}' created with knowledge base tables."
        
        return APIResponse(
            status="success",
            message=message,
            data={
                "config_id": config_id,
                "db_id": user_config.db_id,
                "config_reused": config_was_reused,
                "database_created": not db_existed,
                "database_name": db_name,
                "table_status": table_status
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving user config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save user configuration: {str(e)}"
        )

# # Legacy endpoint for backward compatibility
# @router.post("/user-config/legacy", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
# async def set_user_config_legacy(user_config: UserConfigRequestLegacy):
#     """
#     Legacy endpoint: Create or update user configuration (backward compatibility)
    
#     - **user_id**: Unique identifier for the user
#     - **db_config**: Database connection configuration (legacy format)
#     - **access_level**: Access level (0, 1, or 2)
#     - **accessible_tables**: List of tables the user can access
    
#     Note: This endpoint is deprecated. Use /user-config with db_id instead.
#     """
#     try:
#         db_name = user_config.db_config.DB_NAME
#         db_existed = db_manager.database_exists(db_name)
        
#         # Check if configuration already exists before creating/updating
#         existing_config_id = db_manager.check_config_exists_legacy(user_config)
#         config_was_reused = existing_config_id is not None
        
#         # First check if user's database exists and create it if not
#         db_manager.check_and_create_user_database_legacy(user_config)
        
#         # Then save the user configuration
#         config_id = db_manager.insert_user_config_legacy(user_config)
        
#         # Check table status for better feedback
#         table_status = db_manager.check_knowledge_base_tables_exist(db_name, user_config.db_config)
        
#         # Build appropriate message based on what happened
#         if config_was_reused:
#             if db_existed:
#                 message = f"User configuration for '{user_config.user_id}' reused existing config (ID: {config_id}). Database '{db_name}' already existed. Knowledge base tables ensured."
#             else:
#                 message = f"User configuration for '{user_config.user_id}' reused existing config (ID: {config_id}). Database '{db_name}' created with knowledge base tables."
#         else:
#             if db_existed:
#                 message = f"User configuration for '{user_config.user_id}' created successfully (ID: {config_id}). Database '{db_name}' already existed. Knowledge base tables ensured."
#             else:
#                 message = f"User configuration for '{user_config.user_id}' created successfully (ID: {config_id}). Database '{db_name}' created with knowledge base tables."
        
#         return APIResponse(
#             status="success",
#             message=message + " (Legacy endpoint - consider migrating to /user-config with db_id)",
#             data={
#                 "config_id": config_id,
#                 "config_reused": config_was_reused,
#                 "database_created": not db_existed,
#                 "database_name": db_name,
#                 "table_status": table_status,
#                 "deprecated": True
#             }
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error saving user config (legacy): {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to save user configuration: {str(e)}"
#         )

@router.get("/user-config/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_user_config(user_id: str):
    """
    Retrieve user configuration by user_id
    """
    try:
        config = db_manager.get_user_config(user_id)
        
        if config:
            return APIResponse(
                status="success",
                message="User configuration retrieved successfully",
                data=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User configuration not found for user_id: {user_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user configuration: {str(e)}"
        )



@router.get("/user-config", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_all_user_configs():
    """
    Retrieve all user configurations
    """
    try:
        configs = db_manager.get_all_user_configs()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(configs)} user configurations",
            data={
                "configs": configs,
                "count": len(configs)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving all user configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user configurations: {str(e)}"
        )

@router.get("/user-config/{user_id}/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_user_configs_by_db_endpoint(user_id: str, db_id: int):
    """
    Get all configurations for a specific user and database combination
    
    - **user_id**: The user ID to get configurations for
    - **db_id**: The database configuration ID to filter by
    
    Returns:
    - All configurations for this user-database combination
    - Metadata including count, latest config ID, and database name
    - Error if database configuration doesn't exist
    """
    try:
        # Call the database manager method to get configs for this user-db combination
        result = db_manager.get_configs_by_user_and_db(user_id, db_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {db_id} not found"
            )
        
        return APIResponse(
            status="success",
            message=f"Retrieved {result['count']} configurations for user '{user_id}' with database '{result['database_name']}' (db_id: {db_id})",
            data=result
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting configs for user {user_id} with db_id {db_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user configurations: {str(e)}"
        )

# @router.delete("/user-config/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
# async def delete_user_config(user_id: str):
#     """
#     Delete all user configurations for a user_id
#     """
#     try:
#         # Check if user exists and get all configs
#         existing_configs = db_manager.get_all_configs_for_user(user_id)
#         if not existing_configs:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"User configuration not found for user_id: {user_id}"
#             )
        
#         # Delete all configs for this user
#         conn = db_manager.get_connection()
#         with conn.cursor() as cursor:
#             cursor.execute("DELETE FROM user_config WHERE user_id = %s", (user_id,))
#             deleted_count = cursor.rowcount
#             conn.commit()
#         conn.close()
        
#         return APIResponse(
#             status="success",
#             message=f"All {deleted_count} user configurations for '{user_id}' deleted successfully"
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting user config: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to delete user configuration: {str(e)}"
#         )

# @router.delete("/user-config/{config_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
# async def delete_user_config_by_id(config_id: int):
#     """
#     Delete user configuration by config_id
#     """
#     try:
#         db_manager.delete_config_by_id(config_id)
#         return APIResponse(
#             status="success",
#             message=f"User configuration with config_id {config_id} deleted successfully"
#         )
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting user config by id: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to delete user configuration by id: {str(e)}"
#         )

@router.get("/user-config/{user_id}/all", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_all_configs_for_user_endpoint(user_id: str):
    """
    Get all configurations for a specific user
    """
    try:
        configs = db_manager.get_all_configs_for_user(user_id)
        return APIResponse(
            status="success",
            message=f"Retrieved {len(configs)} configurations for user '{user_id}'",
            data={
                "user_id": user_id,
                "configs": configs,
                "count": len(configs)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all configs for user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get all configurations for user: {str(e)}"
        )

@router.get("/config/{config_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_config_by_id_endpoint(config_id: int):
    """
    Get specific configuration by config_id
    """
    try:
        config = db_manager.get_config_by_id(config_id)
        
        if config:
            return APIResponse(
                status="success",
                message="Configuration retrieved successfully",
                data=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with config_id {config_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving config by id: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )





# @router.post("/config/{config_id}/table-names", response_model=APIResponse, status_code=status.HTTP_200_OK)
# async def append_table_name_endpoint(config_id: int, table_name: str = Body(..., embed=True)):
#     """
#     Append a table name to the table_names list for a specific configuration
    
#     - **config_id**: The configuration ID to append table name to
#     - **table_name**: The table name to append
    
#     Returns:
#     - Success message if table name was appended
#     - Error if configuration doesn't exist or operation fails
#     """
#     try:
#         # Call the database manager method to append table name
#         success = db_manager.append_table_name(config_id, table_name)
        
#         if success:
#             return APIResponse(
#                 status="success",
#                 message=f"Successfully appended table name '{table_name}' to config_id {config_id}",
#                 data={"config_id": config_id, "table_name": table_name, "action": "appended"}
#             )
#         else:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Configuration with config_id {config_id} not found"
#             )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error appending table name '{table_name}' to config_id {config_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to append table name: {str(e)}"
#         )

# @router.delete("/config/{config_id}/table-names/{table_name}", response_model=APIResponse, status_code=status.HTTP_200_OK)
# async def delete_table_name_endpoint(config_id: int, table_name: str):
#     """
#     Delete a specific table name from the table_names list for a configuration
    
#     - **config_id**: The configuration ID to delete table name from
#     - **table_name**: The table name to delete
    
#     Returns:
#     - Success message if table name was deleted
#     - Error if configuration doesn't exist or table name not found
#     """
#     try:
#         # Call the database manager method to delete table name
#         success = db_manager.delete_table_name(config_id, table_name)
        
#         if success:
#             return APIResponse(
#                 status="success",
#                 message=f"Successfully deleted table name '{table_name}' from config_id {config_id}",
#                 data={"config_id": config_id, "table_name": table_name, "action": "deleted"}
#             )
#         else:
#             # Check if config exists first
#             config = db_manager.get_config_by_id(config_id)
#             if not config:
#                 raise HTTPException(
#                     status_code=status.HTTP_404_NOT_FOUND,
#                     detail=f"Configuration with config_id {config_id} not found"
#                 )
#             else:
#                 raise HTTPException(
#                     status_code=status.HTTP_404_NOT_FOUND,
#                     detail=f"Table name '{table_name}' not found in config_id {config_id}"
#                 )
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error deleting table name '{table_name}' from config_id {config_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to delete table name: {str(e)}"
#         )

# User table names endpoints (by user_id instead of config_id)
@router.get("/user/{user_id}/table-names", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_user_table_names_endpoint(user_id: str):
    """
    Get all table names for a user's current configuration
    
    - **user_id**: The user ID to get table names for
    
    Returns:
    - List of table names if found
    - Error if user doesn't exist or any error occurs
    """
    try:
        # Call the database manager method to get table names
        result = db_manager.get_user_table_names(user_id)
        
        if result is None:
            return APIResponse(
                status="success",
                message=f"User configuration not found for user_id: {user_id}",
                data=None
            )
        else:
            return APIResponse(
                status="success",
                message=f"Retrieved {len(result)} table names for user '{user_id}'",
                data=result
            )
            
    except Exception as e:
        logger.error(f"Error getting table names for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get table names for user: {str(e)}"
        )

@router.post("/user/{user_id}/table-names", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def append_user_table_name_endpoint(user_id: str, table_name: str = Body(..., embed=True)):
    """
    Append a table name to the table_names list for a user's current configuration
    
    - **user_id**: The user ID to append table name to
    - **table_name**: The table name to append
    
    Returns:
    - Success message if table name was appended
    - Error if user doesn't exist or operation fails
    """
    try:
        # Call the database manager method to append table name
        success = db_manager.append_user_table_name(user_id, table_name)
        
        if success:
            return APIResponse(
                status="success",
                message=f"Successfully appended table name '{table_name}' for user '{user_id}'",
                data={"user_id": user_id, "table_name": table_name, "action": "appended"}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User configuration not found for user_id: {user_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error appending table name '{table_name}' for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to append table name: {str(e)}"
        )

@router.delete("/user/{user_id}/table-names/{table_name}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def delete_user_table_name_endpoint(user_id: str, table_name: str):
    """
    Delete a specific table name from the table_names list for a user's current configuration
    
    - **user_id**: The user ID to delete table name from
    - **table_name**: The table name to delete
    
    Returns:
    - Success message if table name was deleted
    - Error if user doesn't exist or table name not found
    """
    try:
        # Call the database manager method to delete table name
        success = db_manager.delete_user_table_name(user_id, table_name)
        
        if success:
            return APIResponse(
                status="success",
                message=f"Successfully deleted table name '{table_name}' for user '{user_id}'",
                data={"user_id": user_id, "table_name": table_name, "action": "deleted"}
            )
        else:
            # Check if user exists first
            config = db_manager.get_user_config(user_id)
            if not config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"User configuration not found for user_id: {user_id}"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Table name '{table_name}' not found for user '{user_id}'"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting table name '{table_name}' for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete table name: {str(e)}"
        )

@router.put("/user-config/{config_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def update_user_config_endpoint(config_id: int, update_data: UserConfigUpdateRequest):
    """
    Update user configuration fields for a specific config_id
    
    - **config_id**: The configuration ID to update
    - **update_data**: The data to update (partial update supported)
    
    Updateable fields:
    - **db_id**: Database configuration ID from database_configs table
    - **access_level**: Access level (0, 1, or 2)
    - **accessible_tables**: List of accessible table names
    - **table_names**: List of table names for this configuration
    
    Returns:
    - Success message if configuration was updated
    - Error if configuration doesn't exist or update fails
    """
    try:
        # Call the database manager method to update configuration
        success = db_manager.update_user_config(config_id, update_data)
        
        if success:
            # Get the updated configuration to return in response
            updated_config = db_manager.get_config_by_id(config_id)
            
            return APIResponse(
                status="success",
                message=f"Successfully updated configuration {config_id}",
                data={
                    "config_id": config_id,
                    "updated_config": updated_config,
                    "updated_fields": {
                        "db_id": update_data.db_id is not None,
                        "access_level": update_data.access_level is not None,
                        "accessible_tables": update_data.accessible_tables is not None,
                        "table_names": update_data.table_names is not None
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with config_id {config_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating configuration {config_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )

@router.get("/health", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def data_base_config_health_check():
    """
    Health check endpoint
    """
    try:
        # Test database connection
        conn = db_manager.get_connection()
        conn.close()
        
        return APIResponse(
            status="healthy",
            message="API and database are running",
            data={
                "database": "main_db",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@router.post("/set-user-config", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def set_user_config_endpoint(user_id: str = Body(..., embed=True), config_id: int = Body(..., embed=True)):
    """
    Set a user's current configuration to a specific config_id (make it the latest)
    
    - **user_id**: The user ID to set configuration for
    - **config_id**: The configuration ID to set as current
    
    This endpoint will:
    1. Fetch the configuration details by config_id
    2. Verify the config belongs to the specified user
    3. Mark this configuration as the latest for the user
    4. Ensure the user's database exists with required tables
    
    Returns:
    - Success message with operation details
    - Error if configuration doesn't exist or doesn't belong to user
    """
    try:
        # Call the database manager method to set user config
        result = db_manager.set_user_config_by_config_id(user_id, config_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Configuration with config_id {config_id} not found or does not belong to user {user_id}"
            )
        
        # Build appropriate message based on what happened
        if result['database_created']:
            message = f"Successfully set configuration {config_id} as latest for user '{user_id}'. Database '{result['database_name']}' created with knowledge base tables."
        else:
            message = f"Successfully set configuration {config_id} as latest for user '{user_id}'. Database '{result['database_name']}' already existed. Knowledge base tables ensured."
        
        return APIResponse(
            status="success",
            message=message,
            data=result
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting user config for user {user_id} with config_id {config_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set user configuration: {str(e)}"
        )

@router.get("/database/{db_id}/table-names", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_all_table_names_by_db_id_endpoint(db_id: int):
    """
    Get all table names from all user configurations that use a specific database configuration
    
    - **db_id**: The database configuration ID to get table names for
    
    Returns:
    - Consolidated list of unique table names from all configurations using this database
    - Metadata about which configurations contributed to the result
    - Error if database configuration doesn't exist
    """
    try:
        # Call the database manager method to get all table names for this db_id
        result = db_manager.get_all_table_names_by_db_id(db_id)
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {db_id} not found"
            )
        
        # Safely access the result data
        unique_table_count = result.get('unique_table_count', 0)
        total_configs = result.get('total_configs', 0)
        
        return APIResponse(
            status="success",
            message=f"Retrieved {unique_table_count} unique table names from {total_configs} configurations for database {db_id}",
            data=result
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting table names for database {db_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get table names for database: {str(e)}"
        )

# Database Configs Endpoints
@router.post("/database-config", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_database_config(db_config_request: DatabaseConfigRequest):
    """
    Create new database configuration
    
    - **db_config**: Database connection configuration with DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, and schema
    """
    try:
        db_id = db_manager.insert_database_config(db_config_request)
        
        return APIResponse(
            status="success",
            message=f"Database configuration created successfully with db_id: {db_id}",
            data={
                "db_id": db_id,
                "db_config": db_config_request.db_config.dict()
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating database config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create database configuration: {str(e)}"
        )

@router.get("/database-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_database_config(db_id: int):
    """
    Get database configuration by db_id
    """
    try:
        config = db_manager.get_database_config(db_id)
        
        if config:
            return APIResponse(
                status="success",
                message="Database configuration retrieved successfully",
                data=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration not found for db_id: {db_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving database config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve database configuration: {str(e)}"
        )

@router.get("/database-config", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def get_all_database_configs():
    """
    Get all database configurations
    """
    try:
        configs = db_manager.get_all_database_configs()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(configs)} database configurations",
            data={
                "configs": configs,
                "count": len(configs)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving all database configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve database configurations: {str(e)}"
        )

@router.put("/database-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def update_database_config(db_id: int, update_data: DatabaseConfigUpdateRequest):
    """
    Update database configuration by db_id
    
    - **db_id**: The database configuration ID to update
    - **update_data**: The database configuration data to update
    """
    try:
        success = db_manager.update_database_config(db_id, update_data)
        
        if success:
            # Get the updated configuration to return in response
            updated_config = db_manager.get_database_config(db_id)
            
            return APIResponse(
                status="success",
                message=f"Database configuration {db_id} updated successfully",
                data={
                    "db_id": db_id,
                    "updated_config": updated_config
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {db_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating database config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update database configuration: {str(e)}"
        )

@router.delete("/database-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def delete_database_config(db_id: int):
    """
    Delete database configuration by db_id
    """
    try:
        success = db_manager.delete_database_config(db_id)
        
        if success:
            return APIResponse(
                status="success",
                message=f"Database configuration {db_id} deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with db_id {db_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting database config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete database configuration: {str(e)}"
        )

# if __name__ == '__main__':
#     print("Starting PostgreSQL Database Manager API with FastAPI...")
#     print("Available endpoints:")
#     print("  POST /setup - Setup database and tables")
#     print("  POST /user-config - Create new user configuration (supports multiple DBs per user)")
#     print("  GET /user-config/{user_id} - Get latest user configuration")
#     print("  GET /user-config/{user_id}/all - Get ALL configurations for a user")
#     print("  GET /config/{config_id} - Get specific configuration by config_id")
#     print("  GET /user-config - Get latest configurations for all users")
#     print("  DELETE /user-config/{user_id} - Delete ALL configurations for a user")
#     print("  DELETE /user-config/{config_id} - Delete specific configuration by config_id")
#     print("  GET /user-tables/{user_id} - Get unique table names from user's database")
#     print("  GET /health - Health check")
#     print("\nAPI Documentation:")
#     print("  Swagger UI: http://localhost:8000/docs")
#     print("  ReDoc: http://localhost:8000/redoc")
#     print("\nStarting server on http://localhost:8000")
    
#     uvicorn.run(
#         "data_base_config:app", 
#         host="0.0.0.0", 
#         port=8200, 
#         reload=True,
#         log_level="info"
#     )
app.include_router(router)