import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from fastapi import FastAPI, HTTPException, status, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import sqlite3
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
import sys
import os
from fastapi import FastAPI, Body, APIRouter
import io
import time
import threading
from collections import OrderedDict
import redis
from redis.connection import ConnectionPool
import pickle
# Add the path to table_info_generator and schema_generator
sys.path.append(os.path.join(os.path.dirname(__file__), 'utilites'))
from table_info_generator import generate_table_info
from schema_generator import generate_schema_and_data

def read_schema_file_content(file_path: str) -> str:
    """
    Read the content of a schema file and return it as a string.
    
    Args:
        file_path (str): Path to the schema file
        
    Returns:
        str: The content of the schema file as a string
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        Exception: For other file reading errors
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading schema file {file_path}: {str(e)}")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# REDIS CACHE SYSTEM WITH TTL
# ============================================================================

class RedisCache:
    """
    Redis-based cache with Time-To-Live (TTL) support for high-performance caching.
    Implements response caching with automatic expiration and persistence.
    """
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: str = None, username: str = None, default_ttl: int = 10, max_connections: int = 20):
        """
        Initialize Redis Cache with connection pooling
        
        Args:
            host (str): Redis host (default: localhost)
            port (int): Redis port (default: 6379)
            db (int): Redis database number (default: 0)
            password (str): Redis password (default: None)
            username (str): Redis username for authentication (default: None)
            default_ttl (int): Default TTL in seconds (10 seconds as requested)
            max_connections (int): Maximum connections in pool (default: 20)
        """
        self.default_ttl = default_ttl
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        self._stats_lock = threading.RLock()
        
        try:
            # Create connection pool for better performance with Redis Cloud support
            pool_kwargs = {
                'host': host,
                'port': port,
                'db': db,
                'max_connections': max_connections,
                'retry_on_timeout': True,
                'socket_timeout': 10,  # Increased timeout for cloud connections
                'socket_connect_timeout': 10
            }
            
            # Add authentication if provided
            if password:
                pool_kwargs['password'] = password
            if username:  # Redis Cloud often requires username
                pool_kwargs['username'] = username
            
            self.pool = ConnectionPool(**pool_kwargs)
            
            # Create Redis client with connection pool
            self.redis_client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Redis cache connected successfully to {host}:{port}")
            
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️  Falling back to in-memory cache")
            self.redis_client = None
            self._fallback_cache = OrderedDict()
            self._fallback_lock = threading.RLock()
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            logger.warning("⚠️  Falling back to in-memory cache")
            self.redis_client = None
            self._fallback_cache = OrderedDict()
            self._fallback_lock = threading.RLock()
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from Redis"""
        return pickle.loads(data)
    
    def _increment_stat(self, stat_name: str, count: int = 1):
        """Thread-safe statistics increment"""
        with self._stats_lock:
            self.stats[stat_name] += count
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache if not expired"""
        try:
            if self.redis_client:
                # Redis implementation
                data = self.redis_client.get(key)
                if data is not None:
                    self._increment_stat('hits')
                    return self._deserialize_value(data)
                else:
                    self._increment_stat('misses')
                    return None
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    if key in self._fallback_cache:
                        entry = self._fallback_cache[key]
                        current_time = time.time()
                        
                        if entry['expires_at'] > current_time:
                            # Move to end (LRU behavior)
                            self._fallback_cache.move_to_end(key)
                            self._increment_stat('hits')
                            return entry['value']
                        else:
                            # Expired
                            del self._fallback_cache[key]
                            self._increment_stat('evictions')
                    
                    self._increment_stat('misses')
                    return None
                    
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            self._increment_stat('misses')
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache with TTL"""
        try:
            ttl_seconds = ttl if ttl is not None else self.default_ttl
            
            if self.redis_client:
                # Redis implementation
                serialized_value = self._serialize_value(value)
                success = self.redis_client.setex(key, ttl_seconds, serialized_value)
                if success:
                    self._increment_stat('sets')
                    logger.debug(f"Cached key '{key}' in Redis with TTL {ttl_seconds}s")
                return bool(success)
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    current_time = time.time()
                    expires_at = current_time + ttl_seconds
                    
                    self._fallback_cache[key] = {
                        'value': value,
                        'expires_at': expires_at,
                        'created_at': current_time
                    }
                    
                    # Move to end (LRU behavior)
                    self._fallback_cache.move_to_end(key)
                    
                    # Cleanup expired entries (limit cache size)
                    if len(self._fallback_cache) > 1000:
                        self._cleanup_expired_fallback()
                    
                    self._increment_stat('sets')
                    logger.debug(f"Cached key '{key}' in memory with TTL {ttl_seconds}s")
                    return True
                    
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete specific key from cache"""
        try:
            if self.redis_client:
                # Redis implementation
                deleted = self.redis_client.delete(key)
                if deleted > 0:
                    self._increment_stat('deletes')
                return deleted > 0
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    if key in self._fallback_cache:
                        del self._fallback_cache[key]
                        self._increment_stat('deletes')
                        return True
                    return False
                    
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
            return False
    
    def clear(self) -> int:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                # Get all keys with our cache prefix
                cache_keys = self.redis_client.keys("user_db_*")
                if cache_keys:
                    deleted = self.redis_client.delete(*cache_keys)
                    self._increment_stat('deletes', deleted)
                    return deleted
                return 0
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    count = len(self._fallback_cache)
                    self._fallback_cache.clear()
                    self._increment_stat('deletes', count)
                    return count
                    
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        try:
            if self.redis_client:
                # Redis implementation with pattern matching
                cache_keys = self.redis_client.keys(f"*{pattern}*")
                if cache_keys:
                    deleted = self.redis_client.delete(*cache_keys)
                    self._increment_stat('deletes', deleted)
                    return deleted
                return 0
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    keys_to_delete = [key for key in self._fallback_cache.keys() if pattern in key]
                    count = len(keys_to_delete)
                    
                    for key in keys_to_delete:
                        del self._fallback_cache[key]
                    
                    self._increment_stat('deletes', count)
                    return count
                    
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    def _cleanup_expired_fallback(self):
        """Clean up expired entries in fallback cache"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in list(self._fallback_cache.items()):
            if entry['expires_at'] < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._fallback_cache:
                del self._fallback_cache[key]
                self._increment_stat('evictions')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            cache_info = {
                'cache_type': 'Redis' if self.redis_client else 'In-Memory (Fallback)',
                'connection_status': 'Connected' if self.redis_client else 'Fallback Mode'
            }
            
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info('memory')
                    cache_info.update({
                        'redis_memory_used': redis_info.get('used_memory_human', 'N/A'),
                        'redis_keys_count': self.redis_client.dbsize()
                    })
                except Exception:
                    cache_info['redis_info'] = 'Unable to retrieve'
            else:
                cache_info['fallback_cache_size'] = len(self._fallback_cache)
            
            return {
                **cache_info,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
                'evictions': self.stats['evictions'],
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests,
                'default_ttl_seconds': self.default_ttl
            }
    
    def get_all_keys(self) -> List[str]:
        """Get all cache keys"""
        try:
            if self.redis_client:
                # Redis implementation
                cache_keys = self.redis_client.keys("user_db_*")
                return [key.decode() if isinstance(key, bytes) else str(key) for key in cache_keys]
            else:
                # Fallback to in-memory cache
                with self._fallback_lock:
                    self._cleanup_expired_fallback()
                    return list(self._fallback_cache.keys())
                    
        except Exception as e:
            logger.error(f"Error getting cache keys: {e}")
            return []

# Initialize Redis cache instance with automatic configuration
def initialize_redis_cache() -> RedisCache:
    """Initialize Redis cache with environment-based configuration
    
    Supports both local Redis and Redis Cloud configurations.
    For Redis Cloud, set REDIS_CLOUD=true in environment.
    """
    # Check if Redis Cloud configuration should be used
    use_redis_cloud = os.getenv('REDIS_CLOUD', 'false').lower() == 'true'
    
    if use_redis_cloud:
        # Redis Cloud configuration (as provided by user)
        redis_host = os.getenv('REDIS_HOST', 'redis-18509.c334.asia-southeast2-1.gce.redns.redis-cloud.com')
        redis_port = int(os.getenv('REDIS_PORT', 18509))
        redis_user = os.getenv('REDIS_USER', 'default')
        redis_password = os.getenv('REDIS_PASSWORD', 'LxABurBqAzQJ4zVPzyb1jIduG6sbk02p')
        redis_db = int(os.getenv('REDIS_DB', 0))
        
        logger.info(f"🌩️ Initializing Redis Cloud connection to {redis_host}:{redis_port}")
        
        return RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            username=redis_user,  # Pass Redis Cloud username
            default_ttl=int(os.getenv('CACHE_TTL', 10)),
            max_connections=20
        )
    else:
        # Local Redis configuration (fallback)
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        redis_db = int(os.getenv('REDIS_DB', 0))
        redis_password = os.getenv('REDIS_PASSWORD', None)
        cache_ttl = int(os.getenv('CACHE_TTL', 10))  # 10 seconds default
        
        logger.info(f"🏠 Initializing local Redis connection to {redis_host}:{redis_port}")
        
        return RedisCache(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            default_ttl=cache_ttl,
            max_connections=20
        )

# Initialize global cache instance
user_db_cache = initialize_redis_cache()

# Cache key generators with improved naming
def get_cache_key_user_db_details(user_id: str) -> str:
    """Generate cache key for user database details"""
    return f"user_db_details:v2:{user_id}"

def get_cache_key_user_db_details_lite(user_id: str) -> str:
    """Generate cache key for user database details (lite version with db_schema)"""
    return f"user_db_details_lite:v3:{user_id}"

def get_cache_key_user_db_details_selective(user_id: str, include_table_info: bool, include_db_schema: bool) -> str:
    """Generate cache key for selective user database details"""
    cache_suffix = f"table_{include_table_info}_schema_{include_db_schema}"
    return f"user_db_selective:v2:{user_id}:{cache_suffix}"
# Pydantic Models for request/response validation

# Background Task Models
class TableInfoTaskRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID requesting the task")

class MatchedTablesGenerationRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID requesting the matched tables generation")

class TableInfoTaskResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status: pending, running, completed, failed")
    message: str = Field(..., description="Response message")

class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    user_id: str = Field(..., description="User ID who created the task")
    db_id: int = Field(..., description="Database ID the task is for")
    status: str = Field(..., description="Task status: pending, running, completed, failed")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    result: Optional[str] = Field(None, description="Task result (table info content)")
    error: Optional[str] = Field(None, description="Error message if task failed")
    created_at: str = Field(..., description="Task creation timestamp")
    updated_at: str = Field(..., description="Task last update timestamp")

# Company Management Models
class ParentCompanyCreate(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=255, description="Parent company name")
    description: Optional[str] = Field(default="", description="Company description")
    address: Optional[str] = Field(default="", description="Company address")
    contact_email: Optional[str] = Field(default="", description="Contact email")
    db_id: Optional[int] = Field(None, description="Optional database ID to associate with parent company")
    vector_db_id: Optional[int] = Field(None, description="Optional vector database ID (must reference database_configs.db_id)")

class ParentCompanyUpdate(BaseModel):
    company_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Parent company name")
    description: Optional[str] = Field(None, description="Company description")
    address: Optional[str] = Field(None, description="Company address")
    contact_email: Optional[str] = Field(None, description="Contact email")
    db_id: Optional[int] = Field(None, description="Optional database ID to associate with parent company")
    vector_db_id: Optional[int] = Field(None, description="Optional vector database ID (must reference database_configs.db_id)")

class SubCompanyCreate(BaseModel):
    parent_company_id: int = Field(..., description="Parent company ID")
    company_name: str = Field(..., min_length=1, max_length=255, description="Sub company name")
    description: Optional[str] = Field(default="", description="Company description")
    address: Optional[str] = Field(default="", description="Company address")
    contact_email: Optional[str] = Field(default="", description="Contact email")
    db_id: Optional[int] = Field(None, description="Optional database ID to associate with sub company")
    vector_db_id: Optional[int] = Field(None, description="Optional vector database ID (must reference database_configs.db_id)")

class SubCompanyUpdate(BaseModel):
    company_name: Optional[str] = Field(None, min_length=1, max_length=255, description="Sub company name")
    description: Optional[str] = Field(None, description="Company description")
    address: Optional[str] = Field(None, description="Company address")
    contact_email: Optional[str] = Field(None, description="Contact email")
    db_id: Optional[int] = Field(None, description="Optional database ID to associate with sub company")
    vector_db_id: Optional[int] = Field(None, description="Optional vector database ID (must reference database_configs.db_id)")

# Database Configuration Models
class MSSQLDBConfigCreate(BaseModel):
    db_url: str = Field(..., description="MSSQL database connection URL")
    db_name: str = Field(..., min_length=1, max_length=100, description="Database name")
    business_rule: Optional[str] = Field(default="", description="Business rules for this database")
    table_info: Optional[Dict[str, Any]] = Field(default={}, description="Table information and structure")
    db_schema: Optional[Dict[str, Any]] = Field(default={}, description="Database schema (large JSON)")
    dbPath: Optional[str] = Field(default="", description="Path to database recovery file")
    report_structure: Optional[str] = Field(default="", description="Report structure configuration as text")

class MSSQLDBConfigUpdate(BaseModel):
    db_url: Optional[str] = Field(None, description="MSSQL database connection URL")
    db_name: Optional[str] = Field(None, min_length=1, max_length=100, description="Database name")
    business_rule: Optional[str] = Field(None, description="Business rules for this database")
    table_info: Optional[Dict[str, Any]] = Field(None, description="Table information and structure")
    db_schema: Optional[Dict[str, Any]] = Field(None, description="Database schema (large JSON)")
    dbPath: Optional[str] = Field(None, description="Path to database recovery file")
    report_structure: Optional[str] = Field(None, description="Report structure configuration as text")

class MSSQLDBConfig(BaseModel):
    db_id: str = Field(..., min_length=1, max_length=100, description="Unique database identifier")
    db_url: str = Field(..., description="MSSQL database connection URL")
    db_name: str = Field(..., min_length=1, max_length=100, description="Database name")
    business_rule: Optional[str] = Field(default="", description="Business rules for this database")
    table_info: Optional[Dict[str, Any]] = Field(default={}, description="Table information and structure")
    db_schema: Optional[Dict[str, Any]] = Field(default={}, description="Database schema (large JSON)")
    dbPath: Optional[str] = Field(default="", description="Path to database recovery file")
    report_structure: Optional[str] = Field(default="", description="Report structure configuration as text")

# New models for multipart form data support
# Note: MSSQLDBConfigCreateForm and MSSQLDBConfigUpdateForm are no longer needed
# as we now use individual form fields directly in the endpoints

# Enhanced User Access Models (Define these first)
class DatabaseAccess(BaseModel):
    db_id: int = Field(..., description="Database ID")
    access_level: str = Field(default="full", description="Access level: full, read_only, limited")

class SubCompanyDatabaseAccess(BaseModel):
    sub_company_id: int = Field(..., description="Sub company ID")
    databases: List[DatabaseAccess] = Field(default=[], description="List of databases and access levels")

class UserDatabaseAccess(BaseModel):
    parent_databases: List[DatabaseAccess] = Field(default=[], description="Parent company databases access")
    sub_databases: List[SubCompanyDatabaseAccess] = Field(default=[], description="Sub company databases access")

# User Database Access Vector Model (same structure as UserDatabaseAccess)
class UserDatabaseAccessVector(BaseModel):
    parent_databases: List[DatabaseAccess] = Field(default=[], description="Parent company databases access vector")
    sub_databases: List[SubCompanyDatabaseAccess] = Field(default=[], description="Sub company databases access vector")

# User Access Models (Updated to use IDs with database access)
class UserAccessConfig(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Unique user identifier")
    parent_company_id: int = Field(..., description="Parent company ID")
    sub_company_ids: List[int] = Field(default=[], description="List of sub company IDs this user can access")
    database_access: Optional[UserDatabaseAccess] = Field(default=None, description="Specific database access control")
    database_access_vector: Optional[UserDatabaseAccessVector] = Field(default=None, description="Specific database access vector control")
    table_shows: Optional[Dict[str, List[str]]] = Field(default={}, description="Tables to show for each database")

# Parent Company Database Mapping Models
class ParentCompanyDatabaseCreate(BaseModel):
    db_id: int = Field(..., description="Database ID to bind to parent company")
    access_level: str = Field(default="full", description="Access level: full, read_only, limited")

class ParentCompanyDatabaseUpdate(BaseModel):
    access_level: Optional[str] = Field(None, description="Access level: full, read_only, limited")

# Sub Company Database Mapping Models
class SubCompanyDatabaseCreate(BaseModel):
    db_id: int = Field(..., description="Database ID to bind to sub company")
    access_level: str = Field(default="full", description="Access level: full, read_only, limited")

class SubCompanyDatabaseUpdate(BaseModel):
    access_level: Optional[str] = Field(None, description="Access level: full, read_only, limited")

class UserAccessRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="Unique user identifier")
    parent_company_id: int = Field(..., description="Parent company ID")

class UserAccessUpdate(BaseModel):
    sub_company_ids: List[int] = Field(default=[], description="List of sub company IDs this user can access")
    database_access: Optional[UserDatabaseAccess] = Field(default=None, description="Specific database access control")
    database_access_vector: Optional[UserDatabaseAccessVector] = Field(default=None, description="Specific database access vector control")
    table_shows: Optional[Dict[str, List[str]]] = Field(default={}, description="Tables to show for each database")

class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None

# User Current Database Models
class UserCurrentDBCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255, description="User ID")
    db_id: int = Field(..., description="Database ID to set as current for the user")

class UserCurrentDBUpdate(BaseModel):
    db_id: int = Field(..., description="Database ID to set as current for the user")

class UserCurrentDBResponse(BaseModel):
    user_id: str = Field(..., description="User ID")
    db_id: int = Field(..., description="Current database ID")
    business_rule: Optional[str] = Field(None, description="Business rules for this database")
    table_info: Optional[Dict[str, Any]] = Field(None, description="Table information and structure")
    db_schema: Optional[Dict[str, Any]] = Field(None, description="Database schema")
    created_at: str = Field(..., description="Record creation timestamp")
    updated_at: str = Field(..., description="Record last update timestamp")

class ReportStructureUpdate(BaseModel):
    report_structure: str = Field(..., description="Report structure configuration as text to update")

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
        self.target_db = 'main_db'  # Using the specified database name
        
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
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.default_db,
                user=self.user,
                password=self.password
            )
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
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.default_db,
                user=self.user,
                password=self.password
            )
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
    
    def create_mssql_config_table(self):
        """
        Create mssql_config table for storing database configurations
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS mssql_config (
            db_id SERIAL PRIMARY KEY,
            db_url TEXT NOT NULL,
            db_name VARCHAR(100) NOT NULL,
            business_rule TEXT DEFAULT '',
            table_info JSONB DEFAULT '{}',
            db_schema JSONB DEFAULT '{}',
            dbPath TEXT DEFAULT '',
            report_structure TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_mssql_config_db_name ON mssql_config(db_name);
        CREATE INDEX IF NOT EXISTS idx_mssql_config_created_at ON mssql_config(created_at);
        CREATE INDEX IF NOT EXISTS idx_mssql_config_dbPath ON mssql_config(dbPath);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'mssql_config' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating mssql_config table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create mssql_config table: {str(e)}"
            )

    def create_parent_companies_table(self):
        """
        Create parent_companies table for storing parent company information
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS parent_companies (
            parent_company_id SERIAL PRIMARY KEY,
            company_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT DEFAULT '',
            address TEXT DEFAULT '',
            contact_email VARCHAR(255) DEFAULT '',
            db_id INTEGER REFERENCES mssql_config(db_id) ON DELETE SET NULL,
            vector_db_id INTEGER REFERENCES database_configs(db_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_parent_companies_name ON parent_companies(company_name);
        CREATE INDEX IF NOT EXISTS idx_parent_companies_created_at ON parent_companies(created_at);
        CREATE INDEX IF NOT EXISTS idx_parent_companies_db_id ON parent_companies(db_id);
        CREATE INDEX IF NOT EXISTS idx_parent_companies_vector_db_id ON parent_companies(vector_db_id);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'parent_companies' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating parent_companies table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create parent_companies table: {str(e)}"
            )

    def create_sub_companies_table(self):
        """
        Create sub_companies table for storing sub company information
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sub_companies (
            sub_company_id SERIAL PRIMARY KEY,
            parent_company_id INTEGER NOT NULL REFERENCES parent_companies(parent_company_id) ON DELETE CASCADE,
            company_name VARCHAR(255) NOT NULL,
            description TEXT DEFAULT '',
            address TEXT DEFAULT '',
            contact_email VARCHAR(255) DEFAULT '',
            db_id INTEGER REFERENCES mssql_config(db_id) ON DELETE SET NULL,
            vector_db_id INTEGER REFERENCES database_configs(db_id) ON DELETE SET NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(parent_company_id, company_name)
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_sub_companies_parent_id ON sub_companies(parent_company_id);
        CREATE INDEX IF NOT EXISTS idx_sub_companies_name ON sub_companies(company_name);
        CREATE INDEX IF NOT EXISTS idx_sub_companies_created_at ON sub_companies(created_at);
        CREATE INDEX IF NOT EXISTS idx_sub_companies_db_id ON sub_companies(db_id);
        CREATE INDEX IF NOT EXISTS idx_sub_companies_vector_db_id ON sub_companies(vector_db_id);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'sub_companies' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating sub_companies table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create sub_companies table: {str(e)}"
            )



    def create_user_access_table(self):
        """
        Create user_access table for managing user access to databases (Updated to use IDs)
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_access (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            parent_company_id INTEGER NOT NULL REFERENCES parent_companies(parent_company_id) ON DELETE CASCADE,
            sub_company_ids INTEGER[] DEFAULT ARRAY[]::INTEGER[],
            database_access JSONB DEFAULT '{}',
            database_access_vector JSONB DEFAULT '{}',
            table_shows JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, parent_company_id)
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_user_access_user_id ON user_access(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_access_parent_company_id ON user_access(parent_company_id);
        CREATE INDEX IF NOT EXISTS idx_user_access_created_at ON user_access(created_at);
        CREATE INDEX IF NOT EXISTS idx_user_access_database_access ON user_access USING GIN(database_access);
        CREATE INDEX IF NOT EXISTS idx_user_access_database_access_vector ON user_access USING GIN(database_access_vector);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'user_access' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating user_access table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user_access table: {str(e)}"
            )

    def create_user_current_db_table(self):
        """
        Create user_current_db table for tracking current database selection per user
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_current_db (
            user_id VARCHAR(255) PRIMARY KEY,
            db_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (db_id) REFERENCES mssql_config(db_id) ON DELETE CASCADE
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_user_current_db_user_id ON user_current_db(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_current_db_db_id ON user_current_db(db_id);
        CREATE INDEX IF NOT EXISTS idx_user_current_db_updated_at ON user_current_db(updated_at);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'user_current_db' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating user_current_db table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user_current_db table: {str(e)}"
            )

    def create_user_created_table_tracking_table(self):
        """
        Create the usercreatedtable table in PostgreSQL to track user-created tables.
        This table stores information about tables created by users with their schema details.
        Modified to maintain one row per user_id and db_id combination with business_rule support.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS usercreatedtable (
            id SERIAL PRIMARY KEY,
            db_id INTEGER NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            table_details JSONB NOT NULL DEFAULT '[]',
            business_rule TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (db_id) REFERENCES mssql_config(db_id) ON DELETE CASCADE,
            UNIQUE(user_id, db_id)
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_db_id ON usercreatedtable(db_id);
        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_user_id ON usercreatedtable(user_id);
        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_created_at ON usercreatedtable(created_at);
        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_table_details ON usercreatedtable USING GIN(table_details);
        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_user_db ON usercreatedtable(user_id, db_id);
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'usercreatedtable' created successfully")
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error creating usercreatedtable: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create usercreatedtable: {str(e)}"
            )

    def get_user_table_tracking_record(self, user_id: str, db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the user table tracking record for a specific user and database combination.
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID
            
        Returns:
            Optional[Dict[str, Any]]: Record if exists, None otherwise
        """
        select_query = """
        SELECT id, db_id, user_id, table_details, business_rule, created_at, updated_at
        FROM usercreatedtable
        WHERE user_id = %s AND db_id = %s
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id, db_id))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                record = dict(result)
                record['created_at'] = record['created_at'].isoformat() if record['created_at'] else None
                record['updated_at'] = record['updated_at'].isoformat() if record['updated_at'] else None
                return record
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting user table tracking record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user table tracking record: {str(e)}"
            )

    def create_or_update_user_table_tracking(self, user_id: str, db_id: int, table_details: Dict[str, Any], business_rule: str = "") -> Dict[str, Any]:
        """
        Create or update the user table tracking record for a specific user and database combination.
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID
            table_details (Dict[str, Any]): Table details to add/update
            business_rule (str): Business rule for this user-database combination
            
        Returns:
            Dict[str, Any]: Created/updated record
        """
        try:
            # Check if record exists
            existing_record = self.get_user_table_tracking_record(user_id, db_id)
            
            if existing_record:
                # Update existing record - append new table details to the array
                current_tables = existing_record.get('table_details', [])
                if not isinstance(current_tables, list):
                    current_tables = []
                
                # Add new table details
                current_tables.append(table_details)
                
                update_query = """
                UPDATE usercreatedtable 
                SET table_details = %s, business_rule = %s, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = %s AND db_id = %s
                RETURNING id, db_id, user_id, table_details, business_rule, created_at, updated_at
                """
                
                conn = self.get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(update_query, (json.dumps(current_tables), business_rule, user_id, db_id))
                    result = cursor.fetchone()
                    conn.commit()
                conn.close()
                
                if result:
                    record = dict(result)
                    record['created_at'] = record['created_at'].isoformat() if record['created_at'] else None
                    record['updated_at'] = record['updated_at'].isoformat() if record['updated_at'] else None
                    return record
            else:
                # Create new record
                insert_query = """
                INSERT INTO usercreatedtable (db_id, user_id, table_details, business_rule, created_at, updated_at)
                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                RETURNING id, db_id, user_id, table_details, business_rule, created_at, updated_at
                """
                
                conn = self.get_connection()
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(insert_query, (db_id, user_id, json.dumps([table_details]), business_rule))
                    result = cursor.fetchone()
                    conn.commit()
                conn.close()
                
                if result:
                    record = dict(result)
                    record['created_at'] = record['created_at'].isoformat() if record['created_at'] else None
                    record['updated_at'] = record['updated_at'].isoformat() if record['updated_at'] else None
                    return record
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create or update user table tracking record"
            )
            
        except HTTPException:
            raise
        except psycopg2.Error as e:
            logger.error(f"Error creating/updating user table tracking record: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create/update user table tracking record: {str(e)}"
            )

    def update_user_business_rule(self, user_id: str, db_id: int, business_rule: str) -> Dict[str, Any]:
        """
        Update the business rule for a specific user and database combination.
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID
            business_rule (str): New business rule
            
        Returns:
            Dict[str, Any]: Updated record
        """
        try:
            # Check if record exists
            existing_record = self.get_user_table_tracking_record(user_id, db_id)
            
            if not existing_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No table tracking record found for user '{user_id}' and database '{db_id}'"
                )
            
            update_query = """
            UPDATE usercreatedtable 
            SET business_rule = %s, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = %s AND db_id = %s
            RETURNING id, db_id, user_id, table_details, business_rule, created_at, updated_at
            """
            
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (business_rule, user_id, db_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                record = dict(result)
                record['created_at'] = record['created_at'].isoformat() if record['created_at'] else None
                record['updated_at'] = record['updated_at'].isoformat() if record['updated_at'] else None
                return record
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update business rule"
            )
            
        except HTTPException:
            raise
        except psycopg2.Error as e:
            logger.error(f"Error updating user business rule: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update user business rule: {str(e)}"
            )

    def get_user_business_rule(self, user_id: str, db_id: int) -> Optional[str]:
        """
        Get the business rule for a specific user and database combination.
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID
            
        Returns:
            Optional[str]: Business rule if exists, None otherwise
        """
        try:
            record = self.get_user_table_tracking_record(user_id, db_id)
            if record:
                return record.get('business_rule', '')
            return None
        except Exception as e:
            logger.error(f"Error getting user business rule: {e}")
            return None

    def get_user_tables_by_db(self, user_id: str, db_id: int) -> List[Dict[str, Any]]:
        """
        Get all tables created by a user in a specific database.
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID
            
        Returns:
            List[Dict[str, Any]]: List of table details
        """
        try:
            record = self.get_user_table_tracking_record(user_id, db_id)
            if record:
                table_details = record.get('table_details', [])
                if isinstance(table_details, list):
                    return table_details
                return []
            return []
        except Exception as e:
            logger.error(f"Error getting user tables by database: {e}")
            return []
    
    def setup_database(self):
        """
        Setup complete database structure
        """
        logger.info(f"Starting database setup for '{self.target_db}'...")
        
        # Check if database exists first
        db_existed = self.database_exists(self.target_db)
        if db_existed:
            logger.info(f"✅ Database '{self.target_db}' already exists - will only create missing tables")
        else:
            logger.info(f"🆕 Database '{self.target_db}' does not exist - will create it")
        
        # Create database (safe - won't overwrite if exists)
        self.create_database(self.target_db)
        
        # Create tables (safe - uses IF NOT EXISTS)
        self.create_mssql_config_table()
        self.create_parent_companies_table()
        self.create_sub_companies_table()
        self.create_parent_company_databases_table()
        self.create_sub_company_databases_table()
        self.create_user_access_table()
        self.create_user_current_db_table()
        self.create_user_created_table_tracking_table()
        
        # Create SQLite task management table
        self.create_task_management_table()
        
        # Run migrations for existing databases
        if db_existed:
            self.migrate_add_config_id_columns()
            self.migrate_usercreatedtable_structure()
            self.migrate_add_database_access_vector_column()
            self.migrate_add_report_structure_column()
            logger.info(f"✅ Database setup completed - existing database '{self.target_db}' preserved and migrated")
        else:
            logger.info(f"✅ Database setup completed - new database '{self.target_db}' created")
        return True

    def migrate_add_config_id_columns(self):
        """
        Migrate existing tables to add config_id columns
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Add config_id column to parent_companies if it doesn't exist
                cursor.execute("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'parent_companies' AND column_name = 'config_id'
                        ) THEN
                            ALTER TABLE parent_companies ADD COLUMN config_id VARCHAR(255) DEFAULT NULL;
                            CREATE INDEX IF NOT EXISTS idx_parent_companies_config_id ON parent_companies(config_id);
                        END IF;
                    END $$;
                """)
                
                # Add config_id column to sub_companies if it doesn't exist
                cursor.execute("""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'sub_companies' AND column_name = 'config_id'
                        ) THEN
                            ALTER TABLE sub_companies ADD COLUMN config_id VARCHAR(255) DEFAULT NULL;
                            CREATE INDEX IF NOT EXISTS idx_sub_companies_config_id ON sub_companies(config_id);
                        END IF;
                    END $$;
                """)
                
                conn.commit()
            conn.close()
            logger.info("Successfully migrated tables to add config_id columns")
            return True
        except Exception as e:
            logger.error(f"Error migrating tables: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate tables: {str(e)}"
            )

    def migrate_rename_config_id_to_vector_db_id(self):
        """
        Migrate existing tables to rename config_id columns to vector_db_id
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Rename config_id column to vector_db_id in parent_companies if it exists
                cursor.execute("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'parent_companies' AND column_name = 'config_id'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'parent_companies' AND column_name = 'vector_db_id'
                        ) THEN
                            ALTER TABLE parent_companies RENAME COLUMN config_id TO vector_db_id;
                            DROP INDEX IF EXISTS idx_parent_companies_config_id;
                            CREATE INDEX IF NOT EXISTS idx_parent_companies_vector_db_id ON parent_companies(vector_db_id);
                        END IF;
                    END $$;
                """)
                
                # Rename config_id column to vector_db_id in sub_companies if it exists
                cursor.execute("""
                    DO $$ 
                    BEGIN 
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'sub_companies' AND column_name = 'config_id'
                        ) AND NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = 'sub_companies' AND column_name = 'vector_db_id'
                        ) THEN
                            ALTER TABLE sub_companies RENAME COLUMN config_id TO vector_db_id;
                            DROP INDEX IF EXISTS idx_sub_companies_config_id;
                            CREATE INDEX IF NOT EXISTS idx_sub_companies_vector_db_id ON sub_companies(vector_db_id);
                        END IF;
                    END $$;
                """)
                
                conn.commit()
            conn.close()
            logger.info("Successfully migrated tables to rename config_id to vector_db_id")
            return True
        except Exception as e:
            logger.error(f"Error migrating tables: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate tables: {str(e)}"
            )

    def migrate_vector_db_id_to_integer(self):
        """
        Migrate vector_db_id columns to INTEGER type and add foreign key constraints
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # First, ensure database_configs table exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS database_configs (
                        db_id SERIAL PRIMARY KEY,
                        db_config JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Check if vector_db_id column exists and is VARCHAR
                cursor.execute("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'parent_companies' AND column_name = 'vector_db_id'
                """)
                parent_result = cursor.fetchone()
                
                if parent_result and parent_result[0] == 'character varying':
                    logger.info("Converting parent_companies.vector_db_id from VARCHAR to INTEGER...")
                    
                    # Drop existing index if it exists
                    cursor.execute("DROP INDEX IF EXISTS idx_parent_companies_vector_db_id;")
                    
                    # Convert column to INTEGER
                    cursor.execute("""
                        ALTER TABLE parent_companies 
                        ALTER COLUMN vector_db_id TYPE INTEGER USING 
                        CASE 
                            WHEN vector_db_id IS NULL THEN NULL
                            WHEN vector_db_id = '' THEN NULL
                            ELSE vector_db_id::INTEGER
                        END;
                    """)
                    
                    # Add foreign key constraint
                    cursor.execute("""
                        ALTER TABLE parent_companies 
                        ADD CONSTRAINT fk_parent_companies_vector_db_id 
                        FOREIGN KEY (vector_db_id) REFERENCES database_configs(db_id) ON DELETE SET NULL;
                    """)
                    
                    # Recreate index
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_parent_companies_vector_db_id ON parent_companies(vector_db_id);")
                    
                    logger.info("✅ parent_companies.vector_db_id converted to INTEGER with foreign key")
                
                # Check if vector_db_id column exists and is VARCHAR in sub_companies
                cursor.execute("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'sub_companies' AND column_name = 'vector_db_id'
                """)
                sub_result = cursor.fetchone()
                
                if sub_result and sub_result[0] == 'character varying':
                    logger.info("Converting sub_companies.vector_db_id from VARCHAR to INTEGER...")
                    
                    # Drop existing index if it exists
                    cursor.execute("DROP INDEX IF EXISTS idx_sub_companies_vector_db_id;")
                    
                    # Convert column to INTEGER
                    cursor.execute("""
                        ALTER TABLE sub_companies 
                        ALTER COLUMN vector_db_id TYPE INTEGER USING 
                        CASE 
                            WHEN vector_db_id IS NULL THEN NULL
                            WHEN vector_db_id = '' THEN NULL
                            ELSE vector_db_id::INTEGER
                        END;
                    """)
                    
                    # Add foreign key constraint
                    cursor.execute("""
                        ALTER TABLE sub_companies 
                        ADD CONSTRAINT fk_sub_companies_vector_db_id 
                        FOREIGN KEY (vector_db_id) REFERENCES database_configs(db_id) ON DELETE SET NULL;
                    """)
                    
                    # Recreate index
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_companies_vector_db_id ON sub_companies(vector_db_id);")
                    
                    logger.info("✅ sub_companies.vector_db_id converted to INTEGER with foreign key")
                
                conn.commit()
            conn.close()
            logger.info("Successfully migrated vector_db_id columns to INTEGER with foreign key constraints")
            return True
        except Exception as e:
            logger.error(f"Error migrating vector_db_id to integer: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate vector_db_id to integer: {str(e)}"
            )

    def migrate_usercreatedtable_structure(self):
        """
        Migrate the usercreatedtable to the new structure with business_rule column and unique constraint.
        This handles existing data by consolidating multiple rows per user_id/db_id into single rows.
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if business_rule column exists
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'usercreatedtable' AND column_name = 'business_rule'
                """)
                has_business_rule = cursor.fetchone() is not None
                
                if not has_business_rule:
                    logger.info("🔄 Migrating usercreatedtable to new structure...")
                    
                    # Add business_rule column
                    cursor.execute("ALTER TABLE usercreatedtable ADD COLUMN business_rule TEXT DEFAULT ''")
                    logger.info("✅ Added business_rule column")
                    
                    # Add unique constraint if it doesn't exist
                    cursor.execute("""
                        DO $$ 
                        BEGIN 
                            IF NOT EXISTS (
                                SELECT 1 FROM pg_constraint 
                                WHERE conname = 'usercreatedtable_user_id_db_id_key'
                            ) THEN
                                ALTER TABLE usercreatedtable ADD CONSTRAINT usercreatedtable_user_id_db_id_key 
                                UNIQUE (user_id, db_id);
                            END IF;
                        END $$;
                    """)
                    logger.info("✅ Added unique constraint")
                    
                    # Create new index
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_usercreatedtable_user_db 
                        ON usercreatedtable(user_id, db_id)
                    """)
                    logger.info("✅ Added user_db index")
                    
                    # Consolidate existing data
                    cursor.execute("""
                        WITH consolidated AS (
                            SELECT 
                                db_id,
                                user_id,
                                jsonb_agg(table_details) as table_details,
                                '' as business_rule,
                                MIN(created_at) as created_at,
                                MAX(updated_at) as updated_at
                            FROM usercreatedtable
                            GROUP BY db_id, user_id
                        )
                        DELETE FROM usercreatedtable;
                        
                        INSERT INTO usercreatedtable (db_id, user_id, table_details, business_rule, created_at, updated_at)
                        SELECT db_id, user_id, table_details, business_rule, created_at, updated_at
                        FROM consolidated;
                    """)
                    logger.info("✅ Consolidated existing data")
                    
                    conn.commit()
                    logger.info("✅ Usercreatedtable migration completed successfully")
                else:
                    logger.info("✅ Usercreatedtable already has new structure")
                    
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error during usercreatedtable migration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate usercreatedtable: {str(e)}"
            )
    
    def migrate_add_database_access_vector_column(self):
        """
        Add database_access_vector column to user_access table if it doesn't exist
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if database_access_vector column exists
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'user_access' AND column_name = 'database_access_vector';
                """)
                
                if not cursor.fetchone():
                    logger.info("🔄 Adding database_access_vector column to user_access table...")
                    
                    # Add the new column
                    cursor.execute("""
                        ALTER TABLE user_access 
                        ADD COLUMN database_access_vector JSONB DEFAULT '{}';
                    """)
                    
                    # Create index for the new column
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_user_access_database_access_vector 
                        ON user_access USING GIN(database_access_vector);
                    """)
                    
                    conn.commit()
                    logger.info("✅ database_access_vector column added successfully")
                else:
                    logger.info("✅ database_access_vector column already exists")
                    
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error during database_access_vector migration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate database_access_vector: {str(e)}"
            )

    def migrate_add_report_structure_column(self):
        """
        Add report_structure column to mssql_config table if it doesn't exist
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                # Check if report_structure column exists
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'mssql_config' AND column_name = 'report_structure';
                """)
                
                if not cursor.fetchone():
                    logger.info("🔄 Adding report_structure column to mssql_config table...")
                    
                    # Add the new column
                    cursor.execute("""
                        ALTER TABLE mssql_config 
                        ADD COLUMN report_structure TEXT DEFAULT '';
                    """)
                    
                    # Create index for the new column
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_mssql_config_report_structure 
                        ON mssql_config(report_structure);
                    """)
                    
                    conn.commit()
                    logger.info("✅ report_structure column added successfully")
                else:
                    logger.info("✅ report_structure column already exists")
                    
            conn.close()
            return True
        except psycopg2.Error as e:
            logger.error(f"Error during report_structure migration: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to migrate report_structure: {str(e)}"
            )
    
    def insert_mssql_config(self, config: MSSQLDBConfigCreate) -> Dict[str, Any]:
        """
        Insert new MSSQL configuration with auto-generated db_id
        """
        insert_query = """
        INSERT INTO mssql_config (db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(insert_query, (
                    config.db_url,
                    config.db_name,
                    config.business_rule,
                    json.dumps(config.table_info),
                    json.dumps(config.db_schema),
                    config.dbPath,
                    config.report_structure
                ))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                config_data = dict(result)
                config_data['created_at'] = config_data['created_at'].isoformat() if config_data['created_at'] else None
                config_data['updated_at'] = config_data['updated_at'].isoformat() if config_data['updated_at'] else None
                logger.info(f"MSSQL config created with ID: {config_data['db_id']}")
                return config_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error inserting MSSQL config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save MSSQL configuration: {str(e)}"
            )

    def insert_mssql_config_with_file(self, config: MSSQLDBConfigCreate, file: UploadFile = None) -> Dict[str, Any]:
        """
        Insert new MSSQL database configuration with optional file upload
        """
        try:
            # First create the configuration
            config_data = self.insert_mssql_config(config)
            db_id = config_data['db_id']
            
            # If file is provided, upload it and update the dbPath
            if file:
                file_result = self.upload_database_file(int(db_id), file)
                # Update the config_data with the new file path
                config_data['dbPath'] = file_result['file_path']
                config_data['file_info'] = {
                    'file_path': file_result['file_path'],
                    'file_size': file_result['file_size'],
                    'original_filename': file_result['original_filename']
                }
            
            return config_data
                
        except Exception as e:
            logger.error(f"Error inserting MSSQL config with file: {e}")
            raise

    def update_mssql_config(self, db_id: int, update_data: MSSQLDBConfigUpdate) -> Optional[Dict[str, Any]]:
        """
        Update MSSQL configuration
        """
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if update_data.db_url is not None:
            update_fields.append("db_url = %s")
            update_values.append(update_data.db_url)
        if update_data.db_name is not None:
            update_fields.append("db_name = %s")
            update_values.append(update_data.db_name)
        if update_data.business_rule is not None:
            update_fields.append("business_rule = %s")
            update_values.append(update_data.business_rule)
        if update_data.table_info is not None:
            update_fields.append("table_info = %s")
            update_values.append(json.dumps(update_data.table_info))
        if update_data.db_schema is not None:
            update_fields.append("db_schema = %s")
            update_values.append(json.dumps(update_data.db_schema))
        if update_data.dbPath is not None and update_data.dbPath.strip():
            update_fields.append("dbPath = %s")
            update_values.append(update_data.dbPath.strip())
        if update_data.report_structure is not None:
            update_fields.append("report_structure = %s")
            update_values.append(update_data.report_structure)
        
        if not update_fields:
            return self.get_mssql_config(db_id)
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        update_values.append(db_id)
        
        update_query = f"""
        UPDATE mssql_config 
        SET {', '.join(update_fields)}
        WHERE db_id = %s
        RETURNING db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, update_values)
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                config_data = dict(result)
                config_data['created_at'] = config_data['created_at'].isoformat() if config_data['created_at'] else None
                config_data['updated_at'] = config_data['updated_at'].isoformat() if config_data['updated_at'] else None
                
                # Clear cache for all users who might be using this database
                self._clear_cache_for_database(db_id)
                
                logger.info(f"MSSQL config ID {db_id} updated successfully and cache cleared")
                return config_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error updating MSSQL config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update MSSQL configuration: {str(e)}"
            )

    def update_mssql_config_with_file(self, db_id: int, update_data: MSSQLDBConfigUpdate, file: UploadFile = None) -> Optional[Dict[str, Any]]:
        """
        Update MSSQL configuration with optional file upload
        """
        try:
            # First update the configuration
            config_data = self.update_mssql_config(db_id, update_data)
            
            # If file is provided, upload it and update the dbPath
            if file:
                file_result = self.upload_database_file(db_id, file)
                # Update the config_data with the new file path
                config_data['dbPath'] = file_result['file_path']
                config_data['file_info'] = {
                    'file_path': file_result['file_path'],
                    'file_size': file_result['file_size'],
                    'original_filename': file_result['original_filename']
                }
            
            return config_data
                
        except Exception as e:
            logger.error(f"Error updating MSSQL config with file: {e}")
            raise
    
    def get_mssql_config(self, db_id: int) -> Optional[Dict[str, Any]]:
        """
        Get MSSQL configuration by db_id
        """
        select_query = """
        SELECT db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at
        FROM mssql_config
        WHERE db_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (db_id,))
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
            logger.error(f"Error getting MSSQL config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve MSSQL configuration: {str(e)}"
            )
    
    def get_all_mssql_configs(self) -> List[Dict[str, Any]]:
        """
        Get all MSSQL configurations
        """
        select_query = """
        SELECT db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at
        FROM mssql_config
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
            logger.error(f"Error getting all MSSQL configs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve MSSQL configurations: {str(e)}"
            )

    def delete_mssql_config(self, db_id: int) -> bool:
        """
        Delete MSSQL configuration by db_id
        """
        delete_query = "DELETE FROM mssql_config WHERE db_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (db_id,))
                conn.commit()
            conn.close()
            logger.info(f"MSSQL config ID {db_id} deleted successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting MSSQL config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete MSSQL configuration: {str(e)}"
            )

    def upload_database_file(self, db_id: int, file: UploadFile) -> Dict[str, Any]:
        """
        Upload database recovery file and update dbPath
        
        Files are saved in the structure: uploads/database/{db_id}/{original_name}_{timestamp}{extension}
        """
        try:
            # Validate that the database configuration exists
            db_config = self.get_mssql_config(db_id)
            if not db_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Database configuration with ID {db_id} not found"
                )
            
            # Create directory structure: uploads/database/{db_id}/
            upload_dir = Path(f"uploads/database/{db_id}")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with original name and timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = file.filename or "unknown_file"
            file_path_obj = Path(original_filename)
            name_without_ext = file_path_obj.stem
            file_extension = file_path_obj.suffix
            
            # Format: originalname_timestamp.ext
            safe_filename = f"{name_without_ext}_{timestamp}{file_extension}"
            file_path = upload_dir / safe_filename
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Update the database with the new file path
            update_query = """
            UPDATE mssql_config 
            SET dbPath = %s, updated_at = CURRENT_TIMESTAMP
            WHERE db_id = %s
            RETURNING db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, created_at, updated_at;
            """
            
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (str(file_path), db_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                config_data = dict(result)
                config_data['created_at'] = config_data['created_at'].isoformat() if config_data['created_at'] else None
                config_data['updated_at'] = config_data['updated_at'].isoformat() if config_data['updated_at'] else None
                
                logger.info(f"Database file uploaded for config ID {db_id}: {file_path}")
                return {
                    "message": "Database file uploaded successfully",
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "original_filename": file.filename,
                    "updated_config": config_data
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to update database configuration"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading database file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload database file: {str(e)}"
            )



    def insert_user_access(self, access_config: UserAccessConfig) -> bool:
        """
        Insert or update user access configuration (Updated to use IDs)
        """
        # Validate that parent company exists
        parent_company = self.get_parent_company(access_config.parent_company_id)
        if not parent_company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parent company with ID {access_config.parent_company_id} does not exist"
            )
        
        # Validate that all sub company IDs exist
        for sub_company_id in access_config.sub_company_ids:
            sub_company = self.get_sub_company(sub_company_id)
            if not sub_company:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sub company with ID {sub_company_id} does not exist"
                )
            if sub_company['parent_company_id'] != access_config.parent_company_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sub company with ID {sub_company_id} does not belong to parent company {access_config.parent_company_id}"
                )
        
        upsert_query = """
        INSERT INTO user_access (user_id, parent_company_id, sub_company_ids, database_access, database_access_vector, table_shows, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id, parent_company_id) 
        DO UPDATE SET 
            sub_company_ids = EXCLUDED.sub_company_ids,
            database_access = EXCLUDED.database_access,
            database_access_vector = EXCLUDED.database_access_vector,
            table_shows = EXCLUDED.table_shows,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(upsert_query, (
                    access_config.user_id,
                    access_config.parent_company_id,
                    access_config.sub_company_ids,
                    json.dumps(access_config.database_access.dict() if access_config.database_access else {}),
                    json.dumps(access_config.database_access_vector.dict() if access_config.database_access_vector else {}),
                    json.dumps(access_config.table_shows)
                ))
                conn.commit()
            conn.close()
            logger.info(f"User access config for '{access_config.user_id}' - parent company ID '{access_config.parent_company_id}' inserted/updated successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error inserting user access config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save user access configuration: {str(e)}"
            )
    
    def get_user_access(self, user_id: str, parent_company_id: int) -> Optional[Dict[str, Any]]:
        """
        Get user access configuration by user_id and parent_company_id
        """
        select_query = """
        SELECT ua.user_id, ua.parent_company_id, ua.sub_company_ids, ua.database_access, ua.database_access_vector, ua.table_shows, 
               ua.created_at, ua.updated_at, pc.company_name as parent_company_name
        FROM user_access ua
        JOIN parent_companies pc ON ua.parent_company_id = pc.parent_company_id
        WHERE ua.user_id = %s AND ua.parent_company_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id, parent_company_id))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                # Convert to regular dict and handle datetime serialization
                access = dict(result)
                access['created_at'] = access['created_at'].isoformat() if access['created_at'] else None
                access['updated_at'] = access['updated_at'].isoformat() if access['updated_at'] else None
                return access
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting user access config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user access configuration: {str(e)}"
            )
    
    def get_user_access_by_user_id(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all user access configurations for a specific user
        """
        select_query = """
        SELECT ua.user_id, ua.parent_company_id, ua.sub_company_ids, ua.database_access, ua.database_access_vector, ua.table_shows, 
               ua.created_at, ua.updated_at, pc.company_name as parent_company_name
        FROM user_access ua
        JOIN parent_companies pc ON ua.parent_company_id = pc.parent_company_id
        WHERE ua.user_id = %s
        ORDER BY ua.created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id,))
                results = cursor.fetchall()
            conn.close()
            
            access_configs = []
            for result in results:
                access = dict(result)
                access['created_at'] = access['created_at'].isoformat() if access['created_at'] else None
                access['updated_at'] = access['updated_at'].isoformat() if access['updated_at'] else None
                access_configs.append(access)
            
            return access_configs
        except psycopg2.Error as e:
            logger.error(f"Error getting user access configs by user_id: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user access configurations: {str(e)}"
            )

    def get_all_user_access_configs(self) -> List[Dict[str, Any]]:
        """
        Get all user access configurations
        """
        select_query = """
        SELECT ua.user_id, ua.parent_company_id, ua.sub_company_ids, ua.database_access, ua.database_access_vector, ua.table_shows, 
               ua.created_at, ua.updated_at, pc.company_name as parent_company_name
        FROM user_access ua
        JOIN parent_companies pc ON ua.parent_company_id = pc.parent_company_id
        ORDER BY ua.created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query)
                results = cursor.fetchall()
            conn.close()
            
            access_configs = []
            for result in results:
                access = dict(result)
                access['created_at'] = access['created_at'].isoformat() if access['created_at'] else None
                access['updated_at'] = access['updated_at'].isoformat() if access['updated_at'] else None
                access_configs.append(access)
            
            return access_configs
        except psycopg2.Error as e:
            logger.error(f"Error getting all user access configs: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve user access configurations: {str(e)}"
            )

    def delete_user_access(self, user_id: str, parent_company_id: int) -> bool:
        """
        Delete user access configuration by user_id and parent_company_id
        """
        delete_query = "DELETE FROM user_access WHERE user_id = %s AND parent_company_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (user_id, parent_company_id))
                conn.commit()
            conn.close()
            logger.info(f"User access config for '{user_id}' - parent company ID '{parent_company_id}' deleted successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting user access config: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete user access configuration: {str(e)}"
            )

    def update_user_access_by_user_id(self, user_id: str, update_data: UserAccessUpdate) -> List[Dict[str, Any]]:
        """
        Update user access configuration by user_id
        This will update all access configurations for the user
        """
        try:
            # First, get all existing access configurations for this user
            existing_configs = self.get_user_access_by_user_id(user_id)
            
            if not existing_configs:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No user access configurations found for user_id: {user_id}"
                )
            
            updated_configs = []
            
            # Update each existing configuration
            for existing_config in existing_configs:
                parent_company_id = existing_config['parent_company_id']
                
                # Create updated config with existing parent_company_id but new data
                updated_config = UserAccessConfig(
                    user_id=user_id,
                    parent_company_id=parent_company_id,
                    sub_company_ids=update_data.sub_company_ids,
                    database_access=update_data.database_access,
                    database_access_vector=update_data.database_access_vector,
                    table_shows=update_data.table_shows
                )
                
                # Validate that parent company exists
                parent_company = self.get_parent_company(parent_company_id)
                if not parent_company:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Parent company with ID {parent_company_id} does not exist"
                    )
                
                # Validate that all sub company IDs exist and belong to the parent company
                for sub_company_id in updated_config.sub_company_ids:
                    sub_company = self.get_sub_company(sub_company_id)
                    if not sub_company:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Sub company with ID {sub_company_id} does not exist"
                        )
                    if sub_company['parent_company_id'] != parent_company_id:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Sub company with ID {sub_company_id} does not belong to parent company {parent_company_id}"
                        )
                
                # Validate database access if provided
                if updated_config.database_access:
                    # Validate parent company databases
                    for db_access in updated_config.database_access.parent_databases:
                        if not self.get_mssql_config(db_access.db_id):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Database with db_id {db_access.db_id} does not exist"
                            )
                    
                    # Validate sub company databases
                    for sub_db_access in updated_config.database_access.sub_databases:
                        for db_access in sub_db_access.databases:
                            if not self.get_mssql_config(db_access.db_id):
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Database with db_id {db_access.db_id} does not exist"
                                )
                
                # Validate database access vector if provided
                if updated_config.database_access_vector:
                    # Validate parent company databases
                    for db_access in updated_config.database_access_vector.parent_databases:
                        if not self.get_mssql_config(db_access.db_id):
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Database with db_id {db_access.db_id} does not exist in database_access_vector"
                            )
                    
                    # Validate sub company databases
                    for sub_db_access in updated_config.database_access_vector.sub_databases:
                        for db_access in sub_db_access.databases:
                            if not self.get_mssql_config(db_access.db_id):
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Database with db_id {db_access.db_id} does not exist in database_access_vector"
                                )
                
                # Update the configuration
                self.insert_user_access(updated_config)
                
                # Get the updated configuration
                updated_config_data = self.get_user_access(user_id, parent_company_id)
                if updated_config_data:
                    updated_configs.append(updated_config_data)
            
            logger.info(f"Updated {len(updated_configs)} user access configurations for user '{user_id}'")
            return updated_configs
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating user access by user_id: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update user access configuration: {str(e)}"
            )

    # Parent Company Management Methods
    def create_parent_company(self, company_data: ParentCompanyCreate) -> Dict[str, Any]:
        """
        Create a new parent company
        """
        # Validate db_id if provided
        if company_data.db_id is not None:
            db_config = self.get_mssql_config(company_data.db_id)
            if not db_config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Database with db_id {company_data.db_id} does not exist"
                )
        
        # Validate vector_db_id if provided
        if company_data.vector_db_id is not None:
            if not self.validate_vector_db_id(company_data.vector_db_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Vector database with db_id {company_data.vector_db_id} does not exist in database_configs table"
                )
        
        insert_query = """
        INSERT INTO parent_companies (company_name, description, address, contact_email, db_id, vector_db_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(insert_query, (
                    company_data.company_name,
                    company_data.description,
                    company_data.address,
                    company_data.contact_email,
                    company_data.db_id,
                    company_data.vector_db_id
                ))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                logger.info(f"Parent company '{company_data.company_name}' created with ID: {company['parent_company_id']}")
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error creating parent company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create parent company: {str(e)}"
            )

    def get_parent_company(self, parent_company_id: int) -> Optional[Dict[str, Any]]:
        """
        Get parent company by ID
        """
        select_query = """
        SELECT parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at
        FROM parent_companies
        WHERE parent_company_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (parent_company_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting parent company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve parent company: {str(e)}"
            )

    def get_all_parent_companies(self) -> List[Dict[str, Any]]:
        """
        Get all parent companies
        """
        select_query = """
        SELECT parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at
        FROM parent_companies
        ORDER BY created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query)
                results = cursor.fetchall()
            conn.close()
            
            companies = []
            for result in results:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                companies.append(company)
            
            return companies
        except psycopg2.Error as e:
            logger.error(f"Error getting all parent companies: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve parent companies: {str(e)}"
            )

    def update_parent_company(self, parent_company_id: int, update_data: ParentCompanyUpdate) -> Optional[Dict[str, Any]]:
        """
        Update parent company
        """
        # Validate db_id if provided
        if update_data.db_id is not None:
            db_config = self.get_mssql_config(update_data.db_id)
            if not db_config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Database with db_id {update_data.db_id} does not exist"
                )
        
        # Validate vector_db_id if provided
        if update_data.vector_db_id is not None:
            if not self.validate_vector_db_id(update_data.vector_db_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Vector database with db_id {update_data.vector_db_id} does not exist in database_configs table"
                )
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if update_data.company_name is not None:
            update_fields.append("company_name = %s")
            update_values.append(update_data.company_name)
        if update_data.description is not None:
            update_fields.append("description = %s")
            update_values.append(update_data.description)
        if update_data.address is not None:
            update_fields.append("address = %s")
            update_values.append(update_data.address)
        if update_data.contact_email is not None:
            update_fields.append("contact_email = %s")
            update_values.append(update_data.contact_email)
        if update_data.db_id is not None:
            update_fields.append("db_id = %s")
            update_values.append(update_data.db_id)
        if update_data.vector_db_id is not None:
            update_fields.append("vector_db_id = %s")
            update_values.append(update_data.vector_db_id)
        
        if not update_fields:
            return self.get_parent_company(parent_company_id)
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        update_values.append(parent_company_id)
        
        update_query = f"""
        UPDATE parent_companies 
        SET {', '.join(update_fields)}
        WHERE parent_company_id = %s
        RETURNING parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, update_values)
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                logger.info(f"Parent company ID {parent_company_id} updated successfully")
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error updating parent company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update parent company: {str(e)}"
            )

    def delete_parent_company(self, parent_company_id: int) -> bool:
        """
        Delete parent company (will cascade to sub companies and user access)
        """
        delete_query = "DELETE FROM parent_companies WHERE parent_company_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (parent_company_id,))
                conn.commit()
            conn.close()
            logger.info(f"Parent company ID {parent_company_id} deleted successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting parent company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete parent company: {str(e)}"
            )

    # Sub Company Management Methods
    def create_sub_company(self, company_data: SubCompanyCreate) -> Dict[str, Any]:
        """
        Create a new sub company
        """
        # First verify parent company exists
        parent_company = self.get_parent_company(company_data.parent_company_id)
        if not parent_company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Parent company with ID {company_data.parent_company_id} does not exist"
            )
        
        # Validate db_id if provided
        if company_data.db_id is not None:
            db_config = self.get_mssql_config(company_data.db_id)
            if not db_config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Database with db_id {company_data.db_id} does not exist"
                )
        
        # Validate vector_db_id if provided
        if company_data.vector_db_id is not None:
            if not self.validate_vector_db_id(company_data.vector_db_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Vector database with db_id {company_data.vector_db_id} does not exist in database_configs table"
                )
        
        insert_query = """
        INSERT INTO sub_companies (parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING sub_company_id, parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(insert_query, (
                    company_data.parent_company_id,
                    company_data.company_name,
                    company_data.description,
                    company_data.address,
                    company_data.contact_email,
                    company_data.db_id,
                    company_data.vector_db_id
                ))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                logger.info(f"Sub company '{company_data.company_name}' created with ID: {company['sub_company_id']}")
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error creating sub company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create sub company: {str(e)}"
            )

    def get_sub_company(self, sub_company_id: int) -> Optional[Dict[str, Any]]:
        """
        Get sub company by ID
        """
        select_query = """
        SELECT sc.sub_company_id, sc.parent_company_id, sc.company_name, sc.description, sc.address, sc.contact_email, 
               sc.db_id, sc.vector_db_id, sc.created_at, sc.updated_at, pc.company_name as parent_company_name
        FROM sub_companies sc
        JOIN parent_companies pc ON sc.parent_company_id = pc.parent_company_id
        WHERE sc.sub_company_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (sub_company_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting sub company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve sub company: {str(e)}"
            )

    def get_sub_companies_by_parent(self, parent_company_id: int) -> Dict[str, Any]:
        """
        Get all sub companies for a parent company with enhanced database information
        Returns parent company info, parent databases, and sub companies with their databases
        """
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Get parent company information
                parent_query = """
                SELECT parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at
                FROM parent_companies
                WHERE parent_company_id = %s;
                """
                cursor.execute(parent_query, (parent_company_id,))
                parent_result = cursor.fetchone()
                
                if not parent_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Parent company with ID {parent_company_id} not found"
                    )
                
                parent_company = dict(parent_result)
                parent_company['created_at'] = parent_company['created_at'].isoformat() if parent_company['created_at'] else None
                parent_company['updated_at'] = parent_company['updated_at'].isoformat() if parent_company['updated_at'] else None
                
                # Get parent company's direct database (if any)
                parent_database = None
                if parent_company['db_id']:
                    db_query = """
                    SELECT db_id, db_name, db_url, business_rule, db_schema
                    FROM mssql_config
                    WHERE db_id = %s;
                    """
                    cursor.execute(db_query, (parent_company['db_id'],))
                    db_result = cursor.fetchone()
                    if db_result:
                        parent_database = dict(db_result)
                
                # Get parent company's mapped databases
                parent_mapped_databases_query = """
                SELECT pcd.mapping_id, pcd.db_id, pcd.access_level, pcd.created_at,
                       mc.db_name, mc.db_url, mc.business_rule, mc.db_schema
                FROM parent_company_databases pcd
                JOIN mssql_config mc ON pcd.db_id = mc.db_id
                WHERE pcd.parent_company_id = %s
                ORDER BY pcd.created_at DESC;
                """
                cursor.execute(parent_mapped_databases_query, (parent_company_id,))
                parent_mapped_databases = []
                for row in cursor.fetchall():
                    db_info = dict(row)
                    db_info['created_at'] = db_info['created_at'].isoformat() if db_info['created_at'] else None
                    parent_mapped_databases.append(db_info)
                
                # Get sub companies with their databases
                sub_companies_query = """
                SELECT sc.sub_company_id, sc.parent_company_id, sc.company_name, sc.description, 
                       sc.address, sc.contact_email, sc.db_id, sc.vector_db_id, sc.created_at, sc.updated_at
                FROM sub_companies sc
                WHERE sc.parent_company_id = %s
                ORDER BY sc.created_at DESC;
                """
                cursor.execute(sub_companies_query, (parent_company_id,))
                sub_companies = []
                
                for sub_company_row in cursor.fetchall():
                    sub_company = dict(sub_company_row)
                    sub_company['created_at'] = sub_company['created_at'].isoformat() if sub_company['created_at'] else None
                    sub_company['updated_at'] = sub_company['updated_at'].isoformat() if sub_company['updated_at'] else None
                    
                    # Get sub company's direct database (if any)
                    sub_database = None
                    if sub_company['db_id']:
                        db_query = """
                        SELECT db_id, db_name, db_url, business_rule, db_schema
                        FROM mssql_config
                        WHERE db_id = %s;
                        """
                        cursor.execute(db_query, (sub_company['db_id'],))
                        db_result = cursor.fetchone()
                        if db_result:
                            sub_database = dict(db_result)
                    
                    # Get sub company's mapped databases
                    sub_mapped_databases_query = """
                    SELECT scd.mapping_id, scd.db_id, scd.access_level, scd.created_at,
                           mc.db_name, mc.db_url, mc.business_rule, mc.db_schema
                    FROM sub_company_databases scd
                    JOIN mssql_config mc ON scd.db_id = mc.db_id
                    WHERE scd.sub_company_id = %s
                    ORDER BY scd.created_at DESC;
                    """
                    cursor.execute(sub_mapped_databases_query, (sub_company['sub_company_id'],))
                    sub_mapped_databases = []
                    for row in cursor.fetchall():
                        db_info = dict(row)
                        db_info['created_at'] = db_info['created_at'].isoformat() if db_info['created_at'] else None
                        sub_mapped_databases.append(db_info)
                    
                    # Add database information to sub company
                    sub_company['direct_database'] = sub_database
                    sub_company['mapped_databases'] = sub_mapped_databases
                    sub_companies.append(sub_company)
                
            conn.close()
            
            return {
                "parent_company": parent_company,
                "parent_direct_database": parent_database,
                "parent_mapped_databases": parent_mapped_databases,
                "sub_companies": sub_companies,
                "summary": {
                    "parent_company_id": parent_company_id,
                    "parent_company_name": parent_company['company_name'],
                    "total_sub_companies": len(sub_companies),
                    "parent_has_direct_db": parent_database is not None,
                    "parent_mapped_db_count": len(parent_mapped_databases),
                    "sub_companies_with_direct_db": len([sc for sc in sub_companies if sc['direct_database'] is not None]),
                    "total_sub_mapped_databases": sum(len(sc['mapped_databases']) for sc in sub_companies)
                }
            }
            
        except HTTPException:
            raise
        except psycopg2.Error as e:
            logger.error(f"Error getting sub companies by parent: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve sub companies: {str(e)}"
            )

    def get_all_sub_companies(self) -> List[Dict[str, Any]]:
        """
        Get all sub companies with parent company information
        """
        select_query = """
        SELECT sc.sub_company_id, sc.parent_company_id, sc.company_name, sc.description, sc.address, sc.contact_email, 
               sc.db_id, sc.vector_db_id, sc.created_at, sc.updated_at, pc.company_name as parent_company_name
        FROM sub_companies sc
        JOIN parent_companies pc ON sc.parent_company_id = pc.parent_company_id
        ORDER BY sc.created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query)
                results = cursor.fetchall()
            conn.close()
            
            companies = []
            for result in results:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                companies.append(company)
            
            return companies
        except psycopg2.Error as e:
            logger.error(f"Error getting all sub companies: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve sub companies: {str(e)}"
            )

    def update_sub_company(self, sub_company_id: int, update_data: SubCompanyUpdate) -> Optional[Dict[str, Any]]:
        """
        Update sub company
        """
        # Validate db_id if provided
        if update_data.db_id is not None:
            db_config = self.get_mssql_config(update_data.db_id)
            if not db_config:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Database with db_id {update_data.db_id} does not exist"
                )
        
        # Validate vector_db_id if provided
        if update_data.vector_db_id is not None:
            if not self.validate_vector_db_id(update_data.vector_db_id):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Vector database with db_id {update_data.vector_db_id} does not exist in database_configs table"
                )
        
        # Build dynamic update query
        update_fields = []
        update_values = []
        
        if update_data.company_name is not None:
            update_fields.append("company_name = %s")
            update_values.append(update_data.company_name)
        if update_data.description is not None:
            update_fields.append("description = %s")
            update_values.append(update_data.description)
        if update_data.address is not None:
            update_fields.append("address = %s")
            update_values.append(update_data.address)
        if update_data.contact_email is not None:
            update_fields.append("contact_email = %s")
            update_values.append(update_data.contact_email)
        if update_data.db_id is not None:
            update_fields.append("db_id = %s")
            update_values.append(update_data.db_id)
        if update_data.vector_db_id is not None:
            update_fields.append("vector_db_id = %s")
            update_values.append(update_data.vector_db_id)
        
        if not update_fields:
            return self.get_sub_company(sub_company_id)
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        update_values.append(sub_company_id)
        
        update_query = f"""
        UPDATE sub_companies 
        SET {', '.join(update_fields)}
        WHERE sub_company_id = %s
        RETURNING sub_company_id, parent_company_id, company_name, description, address, contact_email, db_id, vector_db_id, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, update_values)
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                company = dict(result)
                company['created_at'] = company['created_at'].isoformat() if company['created_at'] else None
                company['updated_at'] = company['updated_at'].isoformat() if company['updated_at'] else None
                logger.info(f"Sub company ID {sub_company_id} updated successfully")
                return company
            return None
        except psycopg2.Error as e:
            logger.error(f"Error updating sub company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update sub company: {str(e)}"
            )

    def delete_sub_company(self, sub_company_id: int) -> bool:
        """
        Delete sub company
        """
        delete_query = "DELETE FROM sub_companies WHERE sub_company_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (sub_company_id,))
                conn.commit()
            conn.close()
            logger.info(f"Sub company ID {sub_company_id} deleted successfully")
            return True
        except psycopg2.Error as e:
            logger.error(f"Error deleting sub company: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete sub company: {str(e)}"
            )

    def get_user_databases_with_details(self, user_id: str, parent_company_id: int) -> Dict[str, Any]:
        """
        Get detailed information about databases a user can access (Updated to use IDs)
        """
        try:
            # Get user access configuration
            user_access = self.get_user_access(user_id, parent_company_id)
            if not user_access:
                return {
                    "user_id": user_id,
                    "parent_company_id": parent_company_id,
                    "sub_companies": [],
                    "databases": [],
                    "message": "No access configuration found"
                }
            
            # Get parent company details
            parent_company = self.get_parent_company(parent_company_id)
            
            # Get sub company details
            sub_companies = []
            for sub_company_id in user_access.get('sub_company_ids', []):
                sub_company = self.get_sub_company(sub_company_id)
                if sub_company:
                    sub_companies.append(sub_company)
            
            # Get database mappings for parent company and sub companies
            parent_db_mappings = self.get_company_database_mappings(parent_company_id=parent_company_id)
            sub_db_mappings = []
            
            for sub_id in user_access.get('sub_company_ids', []):
                sub_mappings = self.get_company_database_mappings(sub_company_id=sub_id)
                sub_db_mappings.extend(sub_mappings)
            
            # Combine and deduplicate database mappings
            all_mappings = parent_db_mappings + sub_db_mappings
            unique_databases = {}
            
            for mapping in all_mappings:
                db_id = mapping['db_id']
                if db_id not in unique_databases:
                    db_config = self.get_mssql_config(db_id)
                    if db_config:
                        unique_databases[db_id] = {
                            **db_config,
                            'access_level': mapping['access_level'],
                            'mapping_id': mapping['mapping_id']
                        }
            
            databases = list(unique_databases.values())
            
            return {
                "user_id": user_id,
                "parent_company": parent_company,
                "sub_companies": sub_companies,
                "databases": databases,
                "table_shows": user_access.get('table_shows', {}),
                "access_level": "full" if databases else "none"
            }
        except Exception as e:
            logger.error(f"Error getting user databases with details: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user database details: {str(e)}"
            )

    def create_parent_company_databases_table(self):
        """Create parent company databases mapping table"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS parent_company_databases (
            mapping_id SERIAL PRIMARY KEY,
            parent_company_id INTEGER NOT NULL REFERENCES parent_companies(parent_company_id) ON DELETE CASCADE,
            db_id INTEGER NOT NULL REFERENCES mssql_config(db_id) ON DELETE CASCADE,
            access_level VARCHAR(50) DEFAULT 'full', -- 'full', 'read_only', 'limited'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(parent_company_id, db_id)
        );
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'parent_company_databases' created successfully")
        except Exception as e:
            logger.error(f"Error creating parent_company_databases table: {e}")
            raise

    def create_sub_company_databases_table(self):
        """Create sub company databases mapping table"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS sub_company_databases (
            mapping_id SERIAL PRIMARY KEY,
            sub_company_id INTEGER NOT NULL REFERENCES sub_companies(sub_company_id) ON DELETE CASCADE,
            db_id INTEGER NOT NULL REFERENCES mssql_config(db_id) ON DELETE CASCADE,
            access_level VARCHAR(50) DEFAULT 'full', -- 'full', 'read_only', 'limited'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(sub_company_id, db_id)
        );
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'sub_company_databases' created successfully")
        except Exception as e:
            logger.error(f"Error creating sub_company_databases table: {e}")
            raise

    def create_user_access_table(self):
        """Create user access control table with database access"""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_access (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            parent_company_id INTEGER NOT NULL REFERENCES parent_companies(parent_company_id) ON DELETE CASCADE,
            sub_company_ids INTEGER[] DEFAULT ARRAY[]::INTEGER[],
            database_access JSONB DEFAULT '{}', -- NEW: Specific database access control
            table_shows JSONB DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, parent_company_id)
        );
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
                logger.info("Table 'user_access' created successfully")
        except Exception as e:
            logger.error(f"Error creating user_access table: {e}")
            raise

    def create_parent_company_database(self, parent_company_id: int, mapping_data: ParentCompanyDatabaseCreate) -> Dict[str, Any]:
        """Create parent company database mapping"""
        insert_query = """
        INSERT INTO parent_company_databases (parent_company_id, db_id, access_level, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING mapping_id, parent_company_id, db_id, access_level, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(insert_query, (
                    parent_company_id,
                    mapping_data.db_id,
                    mapping_data.access_level
                ))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                logger.info(f"Parent company database mapping created with ID: {mapping_data['mapping_id']}")
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error creating parent company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create parent company database mapping: {str(e)}"
            )

    def get_parent_company_databases(self, parent_company_id: int) -> List[Dict[str, Any]]:
        """Get all databases for a parent company"""
        select_query = """
        SELECT 
            pcd.mapping_id,
            pcd.parent_company_id,
            pcd.db_id,
            pcd.access_level,
            pcd.created_at,
            pcd.updated_at,
            mc.db_name,
            mc.db_url,
            mc.business_rule,
            mc.table_info,
            mc.db_schema
        FROM parent_company_databases pcd
        JOIN mssql_config mc ON pcd.db_id = mc.db_id
        WHERE pcd.parent_company_id = %s
        ORDER BY pcd.created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (parent_company_id,))
                results = cursor.fetchall()
            conn.close()
            
            databases = []
            for result in results:
                db_data = dict(result)
                db_data['created_at'] = db_data['created_at'].isoformat() if db_data['created_at'] else None
                db_data['updated_at'] = db_data['updated_at'].isoformat() if db_data['updated_at'] else None
                databases.append(db_data)
            
            return databases
        except psycopg2.Error as e:
            logger.error(f"Error getting parent company databases: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get parent company databases: {str(e)}"
            )

    def update_parent_company_database(self, mapping_id: int, update_data: ParentCompanyDatabaseUpdate) -> Optional[Dict[str, Any]]:
        """Update parent company database mapping access level"""
        if update_data.access_level is None:
            return self.get_parent_company_database(mapping_id)
        
        update_query = """
        UPDATE parent_company_databases 
        SET access_level = %s, updated_at = CURRENT_TIMESTAMP
        WHERE mapping_id = %s
        RETURNING mapping_id, parent_company_id, db_id, access_level, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (update_data.access_level, mapping_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                logger.info(f"Parent company database mapping updated: {mapping_id}")
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error updating parent company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update parent company database mapping: {str(e)}"
            )

    def delete_parent_company_database(self, mapping_id: int) -> bool:
        """Delete parent company database mapping"""
        delete_query = "DELETE FROM parent_company_databases WHERE mapping_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (mapping_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Parent company database mapping deleted: {mapping_id}")
            return deleted
        except psycopg2.Error as e:
            logger.error(f"Error deleting parent company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete parent company database mapping: {str(e)}"
            )

    def get_parent_company_database(self, mapping_id: int) -> Optional[Dict[str, Any]]:
        """Get specific parent company database mapping"""
        select_query = """
        SELECT 
            pcd.mapping_id,
            pcd.parent_company_id,
            pcd.db_id,
            pcd.access_level,
            pcd.created_at,
            pcd.updated_at,
            mc.db_name,
            mc.db_url,
            mc.business_rule,
            mc.table_info,
            mc.db_schema
        FROM parent_company_databases pcd
        JOIN mssql_config mc ON pcd.db_id = mc.db_id
        WHERE pcd.mapping_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (mapping_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting parent company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get parent company database mapping: {str(e)}"
            )

    def create_sub_company_database(self, sub_company_id: int, mapping_data: SubCompanyDatabaseCreate) -> Dict[str, Any]:
        """Create sub company database mapping"""
        insert_query = """
        INSERT INTO sub_company_databases (sub_company_id, db_id, access_level, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        RETURNING mapping_id, sub_company_id, db_id, access_level, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(insert_query, (
                    sub_company_id,
                    mapping_data.db_id,
                    mapping_data.access_level
                ))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                logger.info(f"Sub company database mapping created with ID: {mapping_data['mapping_id']}")
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error creating sub company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create sub company database mapping: {str(e)}"
            )

    def get_sub_company_databases(self, sub_company_id: int) -> List[Dict[str, Any]]:
        """Get all databases for a sub company"""
        select_query = """
        SELECT 
            scd.mapping_id,
            scd.sub_company_id,
            scd.db_id,
            scd.access_level,
            scd.created_at,
            scd.updated_at,
            mc.db_name,
            mc.db_url,
            mc.business_rule,
            mc.table_info,
            mc.db_schema
        FROM sub_company_databases scd
        JOIN mssql_config mc ON scd.db_id = mc.db_id
        WHERE scd.sub_company_id = %s
        ORDER BY scd.created_at DESC;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (sub_company_id,))
                results = cursor.fetchall()
            conn.close()
            
            databases = []
            for result in results:
                db_data = dict(result)
                db_data['created_at'] = db_data['created_at'].isoformat() if db_data['created_at'] else None
                db_data['updated_at'] = db_data['updated_at'].isoformat() if db_data['updated_at'] else None
                databases.append(db_data)
            
            return databases
        except psycopg2.Error as e:
            logger.error(f"Error getting sub company databases: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get sub company databases: {str(e)}"
            )

    def update_sub_company_database(self, mapping_id: int, update_data: SubCompanyDatabaseUpdate) -> Optional[Dict[str, Any]]:
        """Update sub company database mapping access level"""
        if update_data.access_level is None:
            return self.get_sub_company_database(mapping_id)
        
        update_query = """
        UPDATE sub_company_databases 
        SET access_level = %s, updated_at = CURRENT_TIMESTAMP
        WHERE mapping_id = %s
        RETURNING mapping_id, sub_company_id, db_id, access_level, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (update_data.access_level, mapping_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                logger.info(f"Sub company database mapping updated: {mapping_id}")
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error updating sub company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update sub company database mapping: {str(e)}"
            )

    def delete_sub_company_database(self, mapping_id: int) -> bool:
        """Delete sub company database mapping"""
        delete_query = "DELETE FROM sub_company_databases WHERE mapping_id = %s;"
        
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute(delete_query, (mapping_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
            conn.close()
            
            if deleted:
                logger.info(f"Sub company database mapping deleted: {mapping_id}")
            return deleted
        except psycopg2.Error as e:
            logger.error(f"Error deleting sub company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete sub company database mapping: {str(e)}"
            )

    def get_sub_company_database(self, mapping_id: int) -> Optional[Dict[str, Any]]:
        """Get specific sub company database mapping"""
        select_query = """
        SELECT 
            scd.mapping_id,
            scd.sub_company_id,
            scd.db_id,
            scd.access_level,
            scd.created_at,
            scd.updated_at,
            mc.db_name,
            mc.db_url,
            mc.business_rule,
            mc.table_info,
            mc.db_schema
        FROM sub_company_databases scd
        JOIN mssql_config mc ON scd.db_id = mc.db_id
        WHERE scd.mapping_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (mapping_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                mapping_data = dict(result)
                mapping_data['created_at'] = mapping_data['created_at'].isoformat() if mapping_data['created_at'] else None
                mapping_data['updated_at'] = mapping_data['updated_at'].isoformat() if mapping_data['updated_at'] else None
                return mapping_data
            return None
        except psycopg2.Error as e:
            logger.error(f"Error getting sub company database mapping: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get sub company database mapping: {str(e)}"
            )

    # ============================================================================
    # SQLITE TASK MANAGEMENT METHODS
    # ============================================================================
    
    def create_task_management_table(self):
        """Create SQLite table for background task management"""
        try:
            # Create SQLite database in the same directory as this file
            db_path = Path(__file__).parent / "task_management.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            create_table_query = """
            CREATE TABLE IF NOT EXISTS background_tasks (
                task_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                db_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
            
            cursor.execute(create_table_query)
            conn.commit()
            conn.close()
            logger.info("Task management table created successfully")
            
        except Exception as e:
            logger.error(f"Error creating task management table: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create task management table: {str(e)}"
            )
    
    def create_table_info_task(self, user_id: str, db_id: int) -> str:
        """Create a new table info generation task"""
        try:
            task_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            
            db_path = Path(__file__).parent / "task_management.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            insert_query = """
            INSERT INTO background_tasks (task_id, user_id, db_id, status, progress, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', 0.0, ?, ?)
            """
            
            cursor.execute(insert_query, (task_id, user_id, db_id, current_time, current_time))
            conn.commit()
            conn.close()
            
            logger.info(f"Created table info task {task_id} for user {user_id} and db {db_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error creating table info task: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create task: {str(e)}"
            )
    
    def update_task_status(self, task_id: str, status: str, progress: float = None, result: str = None, error: str = None):
        """Update task status and progress"""
        try:
            current_time = datetime.now().isoformat()
            
            db_path = Path(__file__).parent / "task_management.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            update_query = """
            UPDATE background_tasks 
            SET status = ?, updated_at = ?
            """
            params = [status, current_time]
            
            if progress is not None:
                update_query += ", progress = ?"
                params.append(progress)
            
            if result is not None:
                update_query += ", result = ?"
                params.append(result)
            
            if error is not None:
                update_query += ", error = ?"
                params.append(error)
            
            update_query += " WHERE task_id = ?"
            params.append(task_id)
            
            cursor.execute(update_query, params)
            conn.commit()
            conn.close()
            
            logger.info(f"Updated task {task_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update task status: {str(e)}"
            )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status by task_id"""
        try:
            db_path = Path(__file__).parent / "task_management.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            select_query = """
            SELECT task_id, user_id, db_id, status, progress, result, error, created_at, updated_at
            FROM background_tasks
            WHERE task_id = ?
            """
            
            cursor.execute(select_query, (task_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'task_id': result[0],
                    'user_id': result[1],
                    'db_id': result[2],
                    'status': result[3],
                    'progress': result[4],
                    'result': result[5],
                    'error': result[6],
                    'created_at': result[7],
                    'updated_at': result[8]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get task status: {str(e)}"
            )

    def update_task_db_id(self, task_id: str, db_id: int) -> bool:
        """
        Update the db_id of an existing task (useful for set-config workflow where db is created during task)
        """
        try:
            db_path = Path(__file__).parent / "task_management.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            current_time = datetime.now().isoformat()
            cursor.execute(
                """
                UPDATE background_tasks
                SET db_id = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (db_id, current_time, task_id),
            )
            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()
            return updated
        except Exception as e:
            logger.error(f"Error updating task db_id: {e}")
            return False
    
    def check_user_database_access(self, user_id: str, db_id: int) -> bool:
        """Check if user has access to the specified database"""
        try:
            # Get user access configurations
            user_access_configs = self.get_user_access_by_user_id(user_id)
            
            if not user_access_configs:
                return False
            
            # Check each access configuration for the database
            for access_config in user_access_configs:
                database_access = access_config.get('database_access')
                if database_access:
                    # Check parent company databases
                    parent_databases = database_access.get('parent_databases', [])
                    for db_access in parent_databases:
                        if db_access.get('db_id') == db_id:
                            return True
                    
                    # Check sub company databases
                    sub_databases = database_access.get('sub_databases', [])
                    for sub_db_access in sub_databases:
                        databases = sub_db_access.get('databases', [])
                        for db_access in databases:
                            if db_access.get('db_id') == db_id:
                                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking user database access: {e}")
            return False

    def validate_vector_db_id(self, vector_db_id: int) -> bool:
        """
        Validate if vector_db_id exists in database_configs table
        """
        try:
            conn = self.get_connection()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 1 FROM database_configs 
                    WHERE db_id = %s
                """, (vector_db_id,))
                
                result = cursor.fetchone()
            conn.close()
            
            return result is not None
        except psycopg2.Error as e:
            logger.error(f"Error validating vector_db_id: {e}")
            return False

    def update_db_schema(self, db_id: int, schema_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update the db_schema column for a specific database configuration
        
        Args:
            db_id (int): Database ID to update
            schema_data (Dict[str, Any]): Schema data to save
            
        Returns:
            Optional[Dict[str, Any]]: Updated configuration or None if not found
        """
        update_query = """
        UPDATE mssql_config 
        SET db_schema = %s, updated_at = CURRENT_TIMESTAMP
        WHERE db_id = %s
        RETURNING db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (json.dumps(schema_data), db_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                config_data = dict(result)
                config_data['created_at'] = config_data['created_at'].isoformat() if config_data['created_at'] else None
                config_data['updated_at'] = config_data['updated_at'].isoformat() if config_data['updated_at'] else None
                logger.info(f"Database schema updated successfully for db_id {db_id}")
                return config_data
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error updating database schema: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update database schema: {str(e)}"
            )

    def update_report_structure(self, db_id: int, report_structure: str) -> Optional[Dict[str, Any]]:
        """
        Update the report_structure column for a specific database configuration
        
        Args:
            db_id (int): Database ID to update
            report_structure (str): Report structure data as text to save
            
        Returns:
            Optional[Dict[str, Any]]: Updated configuration or None if not found
        """
        update_query = """
        UPDATE mssql_config 
        SET report_structure = %s, updated_at = CURRENT_TIMESTAMP
        WHERE db_id = %s
        RETURNING db_id, db_url, db_name, business_rule, table_info, db_schema, dbPath, report_structure, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(update_query, (report_structure, db_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                config_data = dict(result)
                config_data['created_at'] = config_data['created_at'].isoformat() if config_data['created_at'] else None
                config_data['updated_at'] = config_data['updated_at'].isoformat() if config_data['updated_at'] else None
                logger.info(f"Report structure updated successfully for db_id {db_id}")
                return config_data
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error updating report structure: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update report structure: {str(e)}"
            )

    # ============================================================================
    # USER CURRENT DATABASE METHODS
    # ============================================================================
    
    def set_user_current_db(self, user_id: str, db_id: int) -> Dict[str, Any]:
        """
        Set or update the current database for a user (upsert operation)
        
        Args:
            user_id (str): User ID
            db_id (int): Database ID to set as current
            
        Returns:
            Dict[str, Any]: Updated user current database record
        """
        # First validate that the database exists
        db_config = self.get_mssql_config(db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database with ID {db_id} not found"
            )
        
        upsert_query = """
        INSERT INTO user_current_db (user_id, db_id, created_at, updated_at)
        VALUES (%s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id) 
        DO UPDATE SET 
            db_id = EXCLUDED.db_id,
            updated_at = CURRENT_TIMESTAMP
        RETURNING user_id, db_id, created_at, updated_at;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(upsert_query, (user_id, db_id))
                result = cursor.fetchone()
                conn.commit()
            conn.close()
            
            if result:
                record_data = dict(result)
                record_data['created_at'] = record_data['created_at'].isoformat() if record_data['created_at'] else None
                record_data['updated_at'] = record_data['updated_at'].isoformat() if record_data['updated_at'] else None
                logger.info(f"User {user_id} current database set to {db_id}")
                return record_data
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to set user current database"
                )
                
        except psycopg2.Error as e:
            logger.error(f"Error setting user current database: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to set user current database: {str(e)}"
            )
    
    def get_user_current_db(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current database ID for a user
        
        Args:
            user_id (str): User ID
            
        Returns:
            Optional[Dict[str, Any]]: User current database record or None if not found
        """
        select_query = """
        SELECT user_id, db_id, created_at, updated_at
        FROM user_current_db
        WHERE user_id = %s;
        """
        
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(select_query, (user_id,))
                result = cursor.fetchone()
            conn.close()
            
            if result:
                record_data = dict(result)
                record_data['created_at'] = record_data['created_at'].isoformat() if record_data['created_at'] else None
                record_data['updated_at'] = record_data['updated_at'].isoformat() if record_data['updated_at'] else None
                return record_data
            return None
            
        except psycopg2.Error as e:
            logger.error(f"Error getting user current database: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get user current database: {str(e)}"
            )
    
    def get_user_current_db_details(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current database details for a user including business_rule, table_info, db_schema, report_structure.
        Implements automatic Redis caching with TTL for improved performance.
        
        Args:
            user_id (str): User ID
            
        Returns:
            Optional[Dict[str, Any]]: User current database details or None if not found
        """
        # Check cache first (always enabled for performance)
        cache_key = get_cache_key_user_db_details(user_id)
        cached_result = user_db_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"✨ Cache hit for user_db_details: {user_id}")
            return cached_result
        logger.debug(f"🔄 Cache miss for user_db_details: {user_id}")
        
        # First get the current database ID
        current_db_record = self.get_user_current_db(user_id)
        if not current_db_record:
            return None
        
        db_id = current_db_record['db_id']
        
        # Get the full database configuration
        db_config = self.get_mssql_config(db_id)
        if not db_config:
            return None
        
        # Process db_schema to ensure table_name and full_name are included
        db_schema = db_config.get('db_schema', {})
        if db_schema and isinstance(db_schema, dict):
            db_schema = self._process_db_schema_for_response(db_schema)
        
        # Create response data
        result = {
            'user_id': user_id,
            'db_id': db_id,
            'business_rule': db_config.get('business_rule'),
            'table_info': db_config.get('table_info'),
            'db_schema': db_schema,
            'report_structure': db_config.get('report_structure'),
            'created_at': current_db_record['created_at'],
            'updated_at': current_db_record['updated_at']
        }
        
        # Cache the result automatically
        if user_db_cache.set(cache_key, result):
            logger.debug(f"💾 Cached user_db_details for: {user_id}")
        
        return result

    def get_user_current_db_details_lite(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get lightweight current database details for a user (excludes table_info, includes db_schema).
        Implements automatic selective field loading for faster response times.
        
        Args:
            user_id (str): User ID
            
        Returns:
            Optional[Dict[str, Any]]: Lightweight user current database details with db_schema or None if not found
        """
        # Check cache first (always enabled)
        cache_key = get_cache_key_user_db_details_lite(user_id)
        cached_result = user_db_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"✨ Cache hit for user_db_details_lite: {user_id}")
            return cached_result
        logger.debug(f"🔄 Cache miss for user_db_details_lite: {user_id}")
        
        # First get the current database ID
        current_db_record = self.get_user_current_db(user_id)
        if not current_db_record:
            return None
        
        db_id = current_db_record['db_id']
        
        # Get only essential database configuration fields (exclude large fields)  
        db_config = self.get_mssql_config(db_id)
        if not db_config:
            return None
        
        # Process db_schema to ensure table_name and full_name are included
        db_schema = db_config.get('db_schema', {})
        if db_schema and isinstance(db_schema, dict):
            db_schema = self._process_db_schema_for_response(db_schema)
        
        # Create lightweight response data (exclude only table_info, include db_schema as requested)
        result = {
            'user_id': user_id,
            'db_id': db_id,
            'db_name': db_config.get('db_name'),
            'db_url': db_config.get('db_url'),
            'business_rule': db_config.get('business_rule'),
            'db_schema': db_schema,  # Added db_schema as requested
            'report_structure': db_config.get('report_structure'),
            'created_at': current_db_record['created_at'],
            'updated_at': current_db_record['updated_at'],
            'has_table_info': bool(db_config.get('table_info')),
            'has_db_schema': bool(db_config.get('db_schema'))
        }
        
        # Cache the result automatically
        if user_db_cache.set(cache_key, result):
            logger.debug(f"💾 Cached user_db_details_lite for: {user_id}")
        
        return result

    def get_user_current_db_details_selective(self, user_id: str, include_table_info: bool = False, include_db_schema: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get current database details for a user with selective field loading.
        Implements smart caching based on field selection.
        
        Args:
            user_id (str): User ID
            include_table_info (bool): Include table_info field (default: False)
            include_db_schema (bool): Include db_schema field (default: False)
            
        Returns:
            Optional[Dict[str, Any]]: Selective user current database details or None if not found
        """
        # Use smart caching - different cache keys for different field combinations
        cache_key = get_cache_key_user_db_details_selective(user_id, include_table_info, include_db_schema)
        cached_result = user_db_cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"✨ Cache hit for user_db_details_selective: {user_id} (table:{include_table_info}, schema:{include_db_schema})")
            return cached_result
        logger.debug(f"🔄 Cache miss for user_db_details_selective: {user_id}")
        
        # Get the full details first (this might be cached)
        full_details = self.get_user_current_db_details(user_id)
        
        if not full_details:
            return None
        
        # Create selective response based on parameters
        selective_result = {
            'user_id': full_details['user_id'],
            'db_id': full_details['db_id'],
            'business_rule': full_details['business_rule'],
            'report_structure': full_details['report_structure'],
            'created_at': full_details['created_at'],
            'updated_at': full_details['updated_at']
        }
        
        # Conditionally include large fields
        if include_table_info:
            selective_result['table_info'] = full_details['table_info']
        else:
            selective_result['has_table_info'] = bool(full_details.get('table_info'))
        
        if include_db_schema:
            selective_result['db_schema'] = full_details['db_schema']
        else:
            selective_result['has_db_schema'] = bool(full_details.get('db_schema'))
        
        # Add metadata about the selective loading
        selective_result['_metadata'] = {
            'selective_loading': True,
            'included_table_info': include_table_info,
            'included_db_schema': include_db_schema,
            'cache_enabled': True
        }
        
        # Cache the selective result
        if user_db_cache.set(cache_key, selective_result):
            logger.debug(f"💾 Cached user_db_details_selective for: {user_id}")
        
        return selective_result

    def clear_user_cache(self, user_id: str) -> Dict[str, int]:
        """
        Clear all cache entries for a specific user.
        
        Args:
            user_id (str): User ID to clear cache for
            
        Returns:
            Dict[str, int]: Count of cleared entries by type
        """
        cleared_count = {
            'full_details': 0,
            'lite_details': 0,
            'total': 0
        }
        
        # Clear full details cache
        full_key = get_cache_key_user_db_details(user_id)
        if user_db_cache.delete(full_key):
            cleared_count['full_details'] = 1
        
        # Clear lite details cache
        lite_key = get_cache_key_user_db_details_lite(user_id)
        if user_db_cache.delete(lite_key):
            cleared_count['lite_details'] = 1
        
        cleared_count['total'] = cleared_count['full_details'] + cleared_count['lite_details']
        
        logger.info(f"Cleared {cleared_count['total']} cache entries for user: {user_id}")
        return cleared_count

    def _clear_cache_for_database(self, db_id: int) -> int:
        """
        Clear cache entries for all users who might be using this database.
        
        Args:
            db_id (int): Database ID
            
        Returns:
            int: Number of cache entries cleared
        """
        # Since we don't have a reverse mapping from db_id to user_id,
        # we'll clear all cache entries that could be affected
        pattern = "user_db_details"
        cleared_count = user_db_cache.clear_pattern(pattern)
        
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} cache entries for database ID: {db_id}")
        
        return cleared_count

    def _process_db_schema_for_response(self, db_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process db_schema to ensure table_name and full_name are included in matched_tables_details
        and maintain proper field order: schema, table_name, full_name, columns, ...
        
        Args:
            db_schema (Dict[str, Any]): The database schema data
            
        Returns:
            Dict[str, Any]: Processed schema with ensured table_name and full_name fields in correct order
        """
        try:
            if not isinstance(db_schema, dict):
                logger.debug("db_schema is not a dictionary, returning as is")
                return db_schema
            
            # Create a copy to avoid modifying the original
            processed_schema = db_schema.copy()
            
            # Process matched_tables_details if it exists
            if 'matched_tables_details' in processed_schema:
                matched_tables_details = processed_schema['matched_tables_details']
                logger.debug(f"Found {len(matched_tables_details) if isinstance(matched_tables_details, list) else 0} matched tables details")
                
                if isinstance(matched_tables_details, list):
                    processed_details = []
                    for i, table_detail in enumerate(matched_tables_details):
                        if isinstance(table_detail, dict):
                            # Get schema and table_name from the detail
                            schema = table_detail.get('schema', '')
                            table_name = table_detail.get('table_name', '')
                            
                            logger.debug(f"Processing table {i+1}: schema='{schema}', table_name='{table_name}'")
                            
                            # Ensure table_name is present
                            if not table_name and schema:
                                # Try to extract table name from schema if available
                                table_name = table_detail.get('table_name', '')
                            
                            # Ensure full_name is present and correctly formatted
                            if 'full_name' not in table_detail or not table_detail['full_name']:
                                if schema and table_name:
                                    table_detail['full_name'] = f"{schema}.{table_name}"
                                    logger.debug(f"Generated full_name: {table_detail['full_name']}")
                                elif table_name:
                                    table_detail['full_name'] = table_name
                                    logger.debug(f"Generated full_name: {table_detail['full_name']}")
                                else:
                                    table_detail['full_name'] = ''
                                    logger.debug("Generated empty full_name")
                            
                            # Ensure table_name is present
                            if 'table_name' not in table_detail or not table_detail['table_name']:
                                table_detail['table_name'] = table_name
                                logger.debug(f"Set table_name: {table_detail['table_name']}")
                            
                            # Create a new ordered dictionary with the correct field order
                            ordered_detail = {}
                            
                            # Add fields in the desired order: schema, table_name, full_name, then others
                            if 'schema' in table_detail:
                                ordered_detail['schema'] = table_detail['schema']
                            
                            if 'table_name' in table_detail:
                                ordered_detail['table_name'] = table_detail['table_name']
                            
                            if 'full_name' in table_detail:
                                ordered_detail['full_name'] = table_detail['full_name']
                            
                            # Add all other fields in their original order, excluding sample_data
                            for key, value in table_detail.items():
                                if key not in ['schema', 'table_name', 'full_name', 'sample_data']:
                                    ordered_detail[key] = value
                            
                            processed_details.append(ordered_detail)
                        else:
                            logger.debug(f"Table detail {i+1} is not a dictionary, skipping processing")
                            processed_details.append(table_detail)
                    
                    processed_schema['matched_tables_details'] = processed_details
                    logger.debug(f"Processed {len(processed_details)} table details with proper field order")
                else:
                    logger.debug("matched_tables_details is not a list")
            else:
                logger.debug("No matched_tables_details found in db_schema")
            
            return processed_schema
            
        except Exception as e:
            logger.error(f"Error processing db_schema for response: {e}")
            return db_schema


# Initialize database manager with your credentials
db_manager = DatabaseManager(
    host='localhost',
    port=5433,
    user='postgres',
    password='1234'
)

# Background task executor
task_executor = ThreadPoolExecutor(max_workers=3)

async def run_table_info_generation(task_id: str, db_id: int):
    """Background task to generate table info and schema"""
    try:
        logger.info(f"Starting table info and schema generation for task {task_id}")
        
        # Update task status to running
        db_manager.update_task_status(task_id, "running", progress=5.0)
        
        # Get database configuration
        db_config = db_manager.get_mssql_config(db_id)
        if not db_config:
            db_manager.update_task_status(task_id, "failed", error="Database configuration not found")
            return
        
        # Update progress
        db_manager.update_task_status(task_id, "running", progress=15.0)
        
        # Get database URL from config
        db_url = db_config.get('db_url')
        if not db_url:
            db_manager.update_task_status(task_id, "failed", error="Database URL not found in configuration")
            return
        
        # Update progress - both operations will start now
        db_manager.update_task_status(task_id, "running", progress=25.0)
        
        # Run both operations in parallel using asyncio.gather()
        loop = asyncio.get_event_loop()
        logger.info(f"Starting parallel table info and schema generation for task {task_id}")
        
        # Define individual task functions for better error handling
        async def run_table_info_task():
            try:
                logger.info(f"Starting table info generation for task {task_id}")
                result = await loop.run_in_executor(
                    task_executor, 
                    generate_table_info, 
                    db_url
                )
                logger.info(f"Table info generation completed for task {task_id}")
                return {"success": True, "result": result, "error": None}
            except Exception as e:
                logger.error(f"Table info generation failed for task {task_id}: {e}")
                return {"success": False, "result": None, "error": str(e)}
        
        async def run_schema_task():
            try:
                logger.info(f"Starting schema generation for task {task_id}")
                schema_file_path = await loop.run_in_executor(
                    task_executor,
                    generate_schema_and_data,
                    db_url,
                    None,  # output_file=None (will use default)
                    2      # sample_row_count=2 (fixed as requested)
                )
                logger.info(f"Schema generation completed for task {task_id}, file: {schema_file_path}")
                
                # Read the generated schema file content
                logger.info(f"Reading schema file content for task {task_id}")
                schema_content = await loop.run_in_executor(
                    task_executor,
                    read_schema_file_content,
                    schema_file_path
                )
                logger.info(f"Schema file content read successfully for task {task_id}")
                
                return {"success": True, "result": schema_content, "file_path": schema_file_path, "error": None}
            except Exception as e:
                logger.error(f"Schema generation failed for task {task_id}: {e}")
                return {"success": False, "result": None, "file_path": None, "error": str(e)}
        
        # Execute both tasks in parallel
        table_info_result, schema_result = await asyncio.gather(
            run_table_info_task(),
            run_schema_task(),
            return_exceptions=False
        )
        
        # Update progress after both operations complete
        db_manager.update_task_status(task_id, "running", progress=75.0)
        
        # Handle results and check for failures
        table_info = None
        schema_content = None
        schema_file_path = None
        errors = []
        
        if table_info_result["success"]:
            table_info = table_info_result["result"]
        else:
            errors.append(f"Table info generation failed: {table_info_result['error']}")
        
        if schema_result["success"]:
            schema_content = schema_result["result"]  # This is now the file content
            schema_file_path = schema_result["file_path"]  # Keep the path for reference
        else:
            errors.append(f"Schema generation failed: {schema_result['error']}")
        
        # If both operations failed, fail the entire task
        if not table_info and not schema_file:
            error_msg = "Both table info and schema generation failed: " + "; ".join(errors)
            db_manager.update_task_status(task_id, "failed", error=error_msg)
            logger.error(f"Task {task_id} failed: {error_msg}")
            return
        
        # Save table info and schema to the mssql_config table
        try:
            # Convert table_info string to a structured format for storage
            # The table_info is a string from generate_table_info, we'll store it as a dict
            table_info_data = {
                "generated_table_info": table_info,
                "schema": schema_content,  # Store the actual JSON content
                "schema_file_path": schema_file_path,  # Keep the file path for reference
                "generated_at": datetime.now().isoformat(),
                "source": "background_task",
                "task_id": task_id,
                "generation_errors": errors if errors else None  # Track any partial failures
            }
            
            # Update the mssql_config table with the new table_info and schema
            update_data = MSSQLDBConfigUpdate(table_info=table_info_data)
            updated_config = db_manager.update_mssql_config(db_id, update_data)
            
            if updated_config:
                logger.info(f"Table info and schema content saved to mssql_config table for db_id {db_id}")
            else:
                logger.warning(f"Failed to save table info and schema content to mssql_config table for db_id {db_id}")
                
        except Exception as e:
            logger.error(f"Error saving table info and schema content to mssql_config table: {e}")
            # Don't fail the task, just log the error
        
        # Update task with result (include both table info and schema)
        combined_result = {
            "table_info": table_info,
            "schema_content": schema_content,  # The actual JSON content
            "schema_file_path": schema_file_path,  # File path for reference
            "errors": errors if errors else None
        }
        
        # Determine final status based on results
        if errors:
            # Partial success - some operations failed but others succeeded
            status_message = f"Task completed with partial success. Errors: {'; '.join(errors)}"
            logger.warning(f"Task {task_id} completed with partial success: {errors}")
        else:
            # Full success
            status_message = "Task completed successfully"
            logger.info(f"Task {task_id} completed successfully")
        
        db_manager.update_task_status(task_id, "completed", progress=100.0, result=str(combined_result))
        
        logger.info(f"Parallel table info and schema generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Error in parallel table info and schema generation for task {task_id}: {e}")
        db_manager.update_task_status(task_id, "failed", error=str(e))


async def run_set_config_workflow(
    task_id: str,
    db_url: str,
    db_name: str,
    business_rule: str,
    file_bytes: Optional[bytes],
    original_filename: Optional[str],
):
    """Background workflow: create config -> generate table info/schema -> generate matched tables"""
    try:
        logger.info(f"Starting set-config workflow for task {task_id}")
        db_manager.update_task_status(task_id, "running", progress=5.0)

        # Step 1: Create config (without file first)
        config_model = MSSQLDBConfigCreate(
            db_url=db_url,
            db_name=db_name,
            business_rule=business_rule or "",
            table_info={},
            db_schema={},
            dbPath="",
        )
        created_config = db_manager.insert_mssql_config(config_model)
        if not created_config:
            db_manager.update_task_status(task_id, "failed", error="Failed to create configuration")
            return

        db_id = int(created_config["db_id"])  # ensure int
        db_manager.update_task_db_id(task_id, db_id)
        db_manager.update_task_status(task_id, "running", progress=15.0)

        # Step 1b: Upload file if provided
        if file_bytes and original_filename:
            class InMemoryUpload:
                def __init__(self, filename: str, content: bytes):
                    self.filename = filename
                    self.file = io.BytesIO(content)

            upload = InMemoryUpload(original_filename, file_bytes)
            try:
                db_manager.upload_database_file(db_id, upload)  # will update dbPath
            except Exception as e:
                logger.warning(f"File upload failed in workflow (continuing): {e}")
        db_manager.update_task_status(task_id, "running", progress=25.0)

        # If db_url is not provided, skip generation steps and complete the task
        if not db_url:
            final_result = {
                "db_id": db_id,
                "config_created": True,
                "file_uploaded": bool(file_bytes and original_filename),
                "skipped_generation": True,
                "reason": "db_url not provided",
            }
            db_manager.update_task_status(
                task_id, "completed", progress=100.0, result=str(final_result)
            )
            logger.info(f"Set-config workflow completed without generation for task {task_id}")
            return

        # Step 2: Generate table info and schema in parallel
        loop = asyncio.get_event_loop()

        async def table_info_job():
            try:
                result = await loop.run_in_executor(task_executor, generate_table_info, db_url)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        async def schema_job():
            try:
                schema_file_path = await loop.run_in_executor(
                    task_executor, generate_schema_and_data, db_url, None, 2
                )
                schema_content = await loop.run_in_executor(
                    task_executor, read_schema_file_content, schema_file_path
                )
                return {
                    "success": True,
                    "content": schema_content,
                    "file_path": schema_file_path,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        table_info_result, schema_result = await asyncio.gather(
            table_info_job(), schema_job(), return_exceptions=False
        )
        db_manager.update_task_status(task_id, "running", progress=75.0)

        errors: List[str] = []
        table_info = table_info_result.get("result") if table_info_result.get("success") else None
        if not table_info and table_info_result.get("error"):
            errors.append(f"table_info: {table_info_result['error']}")
        schema_content = schema_result.get("content") if schema_result.get("success") else None
        schema_file_path = schema_result.get("file_path") if schema_result.get("success") else None
        if not schema_content and schema_result.get("error"):
            errors.append(f"schema: {schema_result['error']}")

        if not table_info and not schema_content:
            db_manager.update_task_status(
                task_id,
                "failed",
                error="Both table info and schema generation failed: " + "; ".join(errors),
            )
            return

        # Save table info/schema into mssql_config
        try:
            table_info_data = {
                "generated_table_info": table_info,
                "schema": schema_content,
                "schema_file_path": schema_file_path,
                "generated_at": datetime.now().isoformat(),
                "source": "set_config_workflow",
                "task_id": task_id,
                "generation_errors": errors or None,
            }
            db_manager.update_mssql_config(db_id, MSSQLDBConfigUpdate(table_info=table_info_data))
        except Exception as e:
            logger.warning(f"Failed to persist table_info/schema: {e}")

        # Step 3: Generate matched tables and update db_schema
        try:
            from db_manager.utilites.db_schema_to_show import (
                _extract_with_ai,
                _extract_from_schema,
                _get_schema_content,
                _find_matches_with_details,
                _get_unmatched_tables,
            )

            db_config = db_manager.get_mssql_config(db_id)
            business_rules = db_config.get("business_rule", "")
            business_rules_tables: List[str] = []
            if business_rules:
                try:
                    business_rules_tables = _extract_with_ai(business_rules)
                except Exception as e:
                    logger.warning(f"AI extraction failed: {e}")

            schema_tables = _extract_from_schema(db_config)
            schema_value = _get_schema_content(db_config)
            matched_tables, matched_tables_details = _find_matches_with_details(
                business_rules_tables, schema_tables, schema_value
            )

            # Ensure proper ordering and fields
            processed_details = []
            for td in matched_tables_details:
                if isinstance(td, dict):
                    schema = td.get("schema", "")
                    table_name = td.get("table_name", "")
                    if not td.get("full_name"):
                        td["full_name"] = f"{schema}.{table_name}" if schema and table_name else table_name
                    if not td.get("table_name"):
                        td["table_name"] = table_name
                    ordered = {}
                    if "schema" in td:
                        ordered["schema"] = td["schema"]
                    if "table_name" in td:
                        ordered["table_name"] = td["table_name"]
                    if "full_name" in td:
                        ordered["full_name"] = td["full_name"]
                    for k, v in td.items():
                        if k not in ["schema", "table_name", "full_name", "sample_data"]:
                            ordered[k] = v
                    processed_details.append(ordered)
                else:
                    processed_details.append(td)

            matched_tables_result = {
                "status": "success",
                "message": f"Processed database {db_id}",
                "metadata": {
                    "db_id": db_id,
                    "db_name": db_config.get("db_name", ""),
                    "generated_at": datetime.now().isoformat(),
                    "total_business_rules_tables": len(business_rules_tables),
                    "total_schema_tables": len(schema_tables),
                    "total_matches": len(matched_tables),
                },
                "matched_tables": sorted(matched_tables),
                "matched_tables_details": processed_details,
                "business_rules_tables": sorted(business_rules_tables),
                "schema_tables": sorted(schema_tables),
                "unmatched_business_rules": _get_unmatched_tables(business_rules_tables, matched_tables),
                "unmatched_schema": _get_unmatched_tables(schema_tables, matched_tables),
            }

            db_manager.update_db_schema(db_id, matched_tables_result)
        except Exception as e:
            errors.append(f"matched_tables: {str(e)}")

        # Finish
        final_result = {
            "db_id": db_id,
            "config_created": True,
            "file_uploaded": bool(file_bytes and original_filename),
            "errors": errors or None,
        }
        db_manager.update_task_status(
            task_id, "completed", progress=100.0, result=str(final_result)
        )
        logger.info(f"Set-config workflow completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error in set-config workflow for task {task_id}: {e}")
        db_manager.update_task_status(task_id, "failed", error=str(e))


async def run_update_config_workflow(
    task_id: str,
    db_id: int,
    update_model: MSSQLDBConfigUpdate,
    file_bytes: Optional[bytes],
    original_filename: Optional[str],
):
    """Background workflow: update config -> generate table info/schema -> generate matched tables"""
    try:
        logger.info(f"Starting update-config workflow for task {task_id}, db_id {db_id}")
        db_manager.update_task_status(task_id, "running", progress=10.0)

        # Step 1: Update config (without file)
        updated = db_manager.update_mssql_config(db_id, update_model)
        if not updated:
            db_manager.update_task_status(task_id, "failed", error="Configuration not found or not updated")
            return

        # Step 1b: Upload file if provided
        if file_bytes and original_filename:
            class InMemoryUpload:
                def __init__(self, filename: str, content: bytes):
                    self.filename = filename
                    self.file = io.BytesIO(content)

            upload = InMemoryUpload(original_filename, file_bytes)
            try:
                db_manager.upload_database_file(db_id, upload)
            except Exception as e:
                logger.warning(f"File upload failed in update workflow (continuing): {e}")

        db_config = db_manager.get_mssql_config(db_id)
        db_url = db_config.get("db_url") if db_config else None
        if not db_url:
            final_result = {
                "db_id": db_id,
                "config_updated": True,
                "file_uploaded": bool(file_bytes and original_filename),
                "skipped_generation": True,
                "reason": "db_url not available after update",
            }
            db_manager.update_task_status(
                task_id, "completed", progress=100.0, result=str(final_result)
            )
            logger.info(f"Update-config workflow completed without generation for task {task_id}")
            return

        db_manager.update_task_status(task_id, "running", progress=25.0)

        # Step 2: Generate table info and schema in parallel
        loop = asyncio.get_event_loop()

        async def table_info_job():
            try:
                result = await loop.run_in_executor(task_executor, generate_table_info, db_url)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}

        async def schema_job():
            try:
                schema_file_path = await loop.run_in_executor(
                    task_executor, generate_schema_and_data, db_url, None, 2
                )
                schema_content = await loop.run_in_executor(
                    task_executor, read_schema_file_content, schema_file_path
                )
                return {
                    "success": True,
                    "content": schema_content,
                    "file_path": schema_file_path,
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

        table_info_result, schema_result = await asyncio.gather(
            table_info_job(), schema_job(), return_exceptions=False
        )
        db_manager.update_task_status(task_id, "running", progress=75.0)

        errors: List[str] = []
        table_info = table_info_result.get("result") if table_info_result.get("success") else None
        if not table_info and table_info_result.get("error"):
            errors.append(f"table_info: {table_info_result['error']}")
        schema_content = schema_result.get("content") if schema_result.get("success") else None
        schema_file_path = schema_result.get("file_path") if schema_result.get("success") else None
        if not schema_content and schema_result.get("error"):
            errors.append(f"schema: {schema_result['error']}")

        if not table_info and not schema_content:
            db_manager.update_task_status(
                task_id,
                "failed",
                error="Both table info and schema generation failed: " + "; ".join(errors),
            )
            return

        # Save table info/schema into mssql_config
        try:
            table_info_data = {
                "generated_table_info": table_info,
                "schema": schema_content,
                "schema_file_path": schema_file_path,
                "generated_at": datetime.now().isoformat(),
                "source": "update_config_workflow",
                "task_id": task_id,
                "generation_errors": errors or None,
            }
            db_manager.update_mssql_config(db_id, MSSQLDBConfigUpdate(table_info=table_info_data))
        except Exception as e:
            logger.warning(f"Failed to persist table_info/schema (update): {e}")

        # Step 3: Generate matched tables and update db_schema
        try:
            from db_manager.utilites.db_schema_to_show import (
                _extract_with_ai,
                _extract_from_schema,
                _get_schema_content,
                _find_matches_with_details,
                _get_unmatched_tables,
            )

            db_config = db_manager.get_mssql_config(db_id)
            business_rules = db_config.get("business_rule", "")
            business_rules_tables: List[str] = []
            if business_rules:
                try:
                    business_rules_tables = _extract_with_ai(business_rules)
                except Exception as e:
                    logger.warning(f"AI extraction failed: {e}")

            schema_tables = _extract_from_schema(db_config)
            schema_value = _get_schema_content(db_config)
            matched_tables, matched_tables_details = _find_matches_with_details(
                business_rules_tables, schema_tables, schema_value
            )

            processed_details = []
            for td in matched_tables_details:
                if isinstance(td, dict):
                    schema = td.get("schema", "")
                    table_name = td.get("table_name", "")
                    if not td.get("full_name"):
                        td["full_name"] = f"{schema}.{table_name}" if schema and table_name else table_name
                    if not td.get("table_name"):
                        td["table_name"] = table_name
                    ordered = {}
                    if "schema" in td:
                        ordered["schema"] = td["schema"]
                    if "table_name" in td:
                        ordered["table_name"] = td["table_name"]
                    if "full_name" in td:
                        ordered["full_name"] = td["full_name"]
                    for k, v in td.items():
                        if k not in ["schema", "table_name", "full_name", "sample_data"]:
                            ordered[k] = v
                    processed_details.append(ordered)
                else:
                    processed_details.append(td)

            matched_tables_result = {
                "status": "success",
                "message": f"Processed database {db_id}",
                "metadata": {
                    "db_id": db_id,
                    "db_name": db_config.get("db_name", ""),
                    "generated_at": datetime.now().isoformat(),
                    "total_business_rules_tables": len(business_rules_tables),
                    "total_schema_tables": len(schema_tables),
                    "total_matches": len(matched_tables),
                },
                "matched_tables": sorted(matched_tables),
                "matched_tables_details": processed_details,
                "business_rules_tables": sorted(business_rules_tables),
                "schema_tables": sorted(schema_tables),
                "unmatched_business_rules": _get_unmatched_tables(business_rules_tables, matched_tables),
                "unmatched_schema": _get_unmatched_tables(schema_tables, matched_tables),
            }

            db_manager.update_db_schema(db_id, matched_tables_result)
        except Exception as e:
            errors.append(f"matched_tables: {str(e)}")

        final_result = {
            "db_id": db_id,
            "config_updated": True,
            "file_uploaded": bool(file_bytes and original_filename),
            "errors": errors or None,
        }
        db_manager.update_task_status(
            task_id, "completed", progress=100.0, result=str(final_result)
        )
        logger.info(f"Update-config workflow completed for task {task_id}")
    except Exception as e:
        logger.error(f"Error in update-config workflow for task {task_id}: {e}")
        db_manager.update_task_status(task_id, "failed", error=str(e))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for FastAPI application
    """
    # Startup
    try:
        db_manager.setup_database()
        logger.info("Database setup completed on startup")
    except Exception as e:
        logger.warning(f"Could not setup database on startup: {e}")
        logger.info("You can setup the database by calling POST /setup")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

# FastAPI App Setup
app = FastAPI(
    title="MSSQL Configuration Manager API",
    description="API for managing MSSQL database configurations and user access control with company hierarchy",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)



@router.post("/setup", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["System"])
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

@router.post("/migrate", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["System"])
async def migrate_database():
    """
    Run database migrations for existing databases.
    This endpoint handles schema updates and data migrations.
    """
    try:
        migrations_run = []
        
        # Run config_id migration
        try:
            db_manager.migrate_add_config_id_columns()
            migrations_run.append("config_id_columns_added")
        except Exception as e:
            logger.warning(f"Config ID migration failed: {e}")
        
        # Run vector_db_id migration
        try:
            db_manager.migrate_rename_config_id_to_vector_db_id()
            migrations_run.append("vector_db_id_rename_completed")
        except Exception as e:
            logger.warning(f"Vector DB ID migration failed: {e}")
        
        # Run vector_db_id to integer migration
        try:
            db_manager.migrate_vector_db_id_to_integer()
            migrations_run.append("vector_db_id_integer_conversion_completed")
        except Exception as e:
            logger.warning(f"Vector DB ID integer conversion failed: {e}")
        
        # Run usercreatedtable migration
        try:
            db_manager.migrate_usercreatedtable_structure()
            migrations_run.append("usercreatedtable_structure_updated")
        except Exception as e:
            logger.warning(f"Usercreatedtable migration failed: {e}")
        
        # Run database_access_vector migration
        try:
            db_manager.migrate_add_database_access_vector_column()
            migrations_run.append("database_access_vector_column_added")
        except Exception as e:
            logger.warning(f"Database access vector migration failed: {e}")
        
        # Run report_structure migration
        try:
            db_manager.migrate_add_report_structure_column()
            migrations_run.append("report_structure_column_added")
        except Exception as e:
            logger.warning(f"Report structure migration failed: {e}")
        
        return APIResponse(
            status="success",
            message="Database migrations completed successfully",
            data={
                "migrations": migrations_run
            }
        )
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Migration failed: {str(e)}"
        )

# ============================================================================
# MSSQL CONFIGURATION ENDPOINTS
# ============================================================================

@router.post("/mssql-config", response_model=APIResponse, status_code=status.HTTP_201_CREATED, tags=["MSSQL Configuration"])
async def create_mssql_config(
    db_url: str = Form(...),
    db_name: str = Form(...),
    business_rule: str = Form(default=""),
    file: Optional[Union[UploadFile, str]] = File(None)
):
    """
    Create new MSSQL database configuration with optional file upload
    
    - **db_url**: MSSQL database connection URL
    - **db_name**: Database name
    - **business_rule**: Business rules for this database (optional)
    - **file**: Optional database recovery file (e.g., .bak, .sql, .mdf, etc.)
    
    Supports both JSON configuration and file upload in a single request.
    dbPath will be set automatically when a file is uploaded.
    table_info and db_schema will be generated automatically via background tasks.
    
    **Note**: This endpoint accepts multipart/form-data. For JSON-only requests without file upload,
    use POST /mssql-config/json instead.
    """
    try:
        # Create the config model (table_info and db_schema will be generated automatically)
        config_model = MSSQLDBConfigCreate(
            db_url=db_url,
            db_name=db_name,
            business_rule=business_rule,
            table_info={},  # Will be generated automatically
            db_schema={},   # Will be generated automatically
            dbPath=""       # Will be set automatically if file is uploaded
        )
        
        # Validate file if provided
        if isinstance(file, UploadFile) and file.filename:
            allowed_extensions = ['.bak', '.sql', '.mdf', '.ldf', '.trn', '.dmp', '.dump']
            file_extension = Path(file.filename).suffix.lower()
            if file_extension and file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
                )
        
        # Create configuration with optional file upload
        safe_file = file if (isinstance(file, UploadFile) and file.filename) else None
        config_data = db_manager.insert_mssql_config_with_file(config_model, safe_file)
        
        if config_data:
            message = f"MSSQL configuration created successfully with ID: {config_data['db_id']}"
            if isinstance(file, UploadFile) and file.filename:
                message += f" and file uploaded: {file.filename}"
            
            return APIResponse(
                status="success",
                message=message,
                data=config_data
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create MSSQL configuration"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating MSSQL config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create MSSQL configuration: {str(e)}"
        )

@router.get("/mssql-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def get_mssql_config(db_id: int):
    """
    Retrieve MSSQL configuration by db_id
    """
    try:
        config = db_manager.get_mssql_config(db_id)
        
        if config:
            return APIResponse(
                status="success",
                message="MSSQL configuration retrieved successfully",
                data=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MSSQL configuration not found for db_id: {db_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving MSSQL config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve MSSQL configuration: {str(e)}"
        )

@router.get("/mssql-config", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def get_all_mssql_configs():
    """
    Retrieve all MSSQL configurations
    """
    try:
        configs = db_manager.get_all_mssql_configs()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(configs)} MSSQL configurations",
            data={
                "configs": configs,
                "count": len(configs)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving all MSSQL configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve MSSQL configurations: {str(e)}"
        )

@router.put("/mssql-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def update_mssql_config(
    db_id: int,
    db_url: str = Form(None),
    db_name: str = Form(None),
    business_rule: str = Form(None),
    file: Optional[Union[UploadFile, str]] = File(None)
):
    """
    Update MSSQL configuration with optional file upload
    
    - **db_url**: Updated MSSQL database connection URL (optional)
    - **db_name**: Updated database name (optional)
    - **business_rule**: Updated business rules (optional)
    - **file**: Optional database recovery file to upload (e.g., .bak, .sql, .mdf, etc.)
    
    Supports both JSON configuration updates and file upload in a single request.
    dbPath will be set automatically when a file is uploaded.
    table_info and db_schema will be generated automatically via background tasks.
    
    **Note**: This endpoint accepts multipart/form-data. For JSON-only requests without file upload,
    use PUT /mssql-config/{db_id}/json instead.
    """
    try:
        # Build update data dictionary - only include non-empty fields
        update_data = {}
        
        if db_url is not None and db_url.strip():
            update_data['db_url'] = db_url.strip()
            logger.info(f"Update endpoint: Including db_url field")
        if db_name is not None and db_name.strip():
            update_data['db_name'] = db_name.strip()
            logger.info(f"Update endpoint: Including db_name field")
        if business_rule is not None and business_rule.strip():
            update_data['business_rule'] = business_rule.strip()
            logger.info(f"Update endpoint: Including business_rule field")
        
        # Create update model
        update_model = MSSQLDBConfigUpdate(**update_data)
        
        # Validate file if provided
        if isinstance(file, UploadFile) and file.filename:
            allowed_extensions = ['.bak', '.sql', '.mdf', '.ldf', '.trn', '.dmp', '.dump']
            file_extension = Path(file.filename).suffix.lower()
            if file_extension and file_extension not in allowed_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
                )
        
        # Update configuration with optional file upload
        safe_file = file if (isinstance(file, UploadFile) and file.filename) else None
        config = db_manager.update_mssql_config_with_file(db_id, update_model, safe_file)
        
        if config:
            message = f"MSSQL configuration ID {db_id} updated successfully"
            if isinstance(file, UploadFile) and file.filename:
                message += f" and file uploaded: {file.filename}"
            
            return APIResponse(
                status="success",
                message=message,
                data=config
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MSSQL configuration with ID {db_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MSSQL config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update MSSQL configuration: {str(e)}"
        )

@router.delete("/mssql-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def delete_mssql_config(db_id: int):
    """
    Delete MSSQL configuration by db_id
    """
    try:
        # Check if config exists
        existing_config = db_manager.get_mssql_config(db_id)
        if not existing_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MSSQL configuration not found for db_id: {db_id}"
            )
        
        db_manager.delete_mssql_config(db_id)
        
        return APIResponse(
            status="success",
            message=f"MSSQL configuration ID {db_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MSSQL config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete MSSQL configuration: {str(e)}"
        )



@router.post("/mssql-config/{db_id}/generate-table-info", response_model=APIResponse, status_code=status.HTTP_202_ACCEPTED, tags=["MSSQL Configuration"])
async def start_table_info_generation(db_id: int, task_request: TableInfoTaskRequest):
    """
    Start background parallel table info and schema generation for a specific database
    
    - **db_id**: ID of the MSSQL configuration to generate table info and schema for (from URL path)
    - **user_id**: User ID requesting the task (from request body, for access control)
    
    This endpoint starts a background task that generates both table info and schema with 2 sample rows in parallel.
    Both operations run simultaneously for faster execution. Returns immediately with a task ID. 
    Use GET /mssql-config/tasks/{task_id} to check the task status.
    """
    try:
        # Verify database exists
        db_config = db_manager.get_mssql_config(db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with ID {db_id} not found"
            )
        
        # Check user access to this database
        if not db_manager.check_user_database_access(task_request.user_id, db_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User {task_request.user_id} does not have access to database {db_id}"
            )
        
        # Create task record
        task_id = db_manager.create_table_info_task(task_request.user_id, db_id)
        
        # Start background task
        asyncio.create_task(run_table_info_generation(task_id, db_id))
        
        return APIResponse(
            status="accepted",
            message=f"Parallel table info and schema generation task started for database {db_id}",
            data={
                "task_id": task_id,
                "status": "pending",
                "db_id": db_id,
                "user_id": task_request.user_id
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting table info generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start parallel table info and schema generation: {str(e)}"
        )

@router.get("/mssql-config/tasks/{task_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def get_table_info_task_status(task_id: str):
    """
    Get the status of a parallel table info and schema generation task
    
    - **task_id**: Unique task identifier returned by the start endpoint
    
    Returns task status including progress, result (containing both table info and schema), and any errors.
    Both operations run in parallel for faster execution.
    """
    try:
        task_status = db_manager.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task with ID {task_id} not found"
            )
        
        return APIResponse(
            status="success",
            message="Task status retrieved successfully",
            data=task_status
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.post("/mssql-config/{db_id}/generate-matched-tables", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def generate_matched_tables_and_update_schema(db_id: int, request: MatchedTablesGenerationRequest):
    """
    Generate matched tables details using business rules and database schema, then save to db_schema column
    
    - **db_id**: Database ID to process
    - **user_id**: User ID requesting the operation (from request body)
    
    This endpoint:
    1. Validates user access to the database
    2. Gets database configuration directly from database
    3. Extracts and matches table names from business rules and schema
    4. Updates the db_schema column in mssql_config table with the result
    5. Returns the generated matched tables data
    
    The matched tables data includes:
    - Matched table names between business rules and schema
    - Detailed schema information for matched tables
    - Business rules tables and schema tables
    - Unmatched tables from both sources
    """
    try:
        user_id = request.user_id
        
        # Step 1: Validate user access to the database
        if not db_manager.check_user_database_access(user_id, db_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User {user_id} does not have access to database {db_id}"
            )
        
        # Step 2: Get database configuration directly from database
        db_config = db_manager.get_mssql_config(db_id)
        if not db_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Database configuration with ID {db_id} not found"
            )
        
        # Step 3: Extract table names from business rules using AI
        business_rules = db_config.get("business_rule", "")
        business_rules_tables = []
        
        if business_rules:
            # Import the AI extraction function
            from db_manager.utilites.db_schema_to_show import _extract_with_ai
            business_rules_tables = _extract_with_ai(business_rules)
            logger.info(f"Extracted {len(business_rules_tables)} tables from business rules")
        else:
            logger.warning("No business rules found")
        
        # Step 4: Extract table names and schema content
        from db_manager.utilites.db_schema_to_show import _extract_from_schema, _get_schema_content
        schema_tables = _extract_from_schema(db_config)
        schema_content = _get_schema_content(db_config)
        logger.info(f"Extracted {len(schema_tables)} tables from schema")
        
        # Step 5: Find matches and get detailed schema info
        from db_manager.utilites.db_schema_to_show import _find_matches_with_details, _get_unmatched_tables
        matched_tables, matched_tables_details = _find_matches_with_details(
            business_rules_tables, schema_tables, schema_content
        )
        
        # Step 5.5: Ensure matched_tables_details have proper table_name and full_name fields in correct order
        processed_matched_tables_details = []
        for table_detail in matched_tables_details:
            if isinstance(table_detail, dict):
                # Get schema and table_name from the detail
                schema = table_detail.get('schema', '')
                table_name = table_detail.get('table_name', '')
                
                # Ensure full_name is present and correctly formatted
                if 'full_name' not in table_detail or not table_detail['full_name']:
                    if schema and table_name:
                        table_detail['full_name'] = f"{schema}.{table_name}"
                    elif table_name:
                        table_detail['full_name'] = table_name
                    else:
                        table_detail['full_name'] = ''
                
                # Ensure table_name is present
                if 'table_name' not in table_detail or not table_detail['table_name']:
                    table_detail['table_name'] = table_name
                
                # Create a new ordered dictionary with the correct field order
                ordered_detail = {}
                
                # Add fields in the desired order: schema, table_name, full_name, then others
                if 'schema' in table_detail:
                    ordered_detail['schema'] = table_detail['schema']
                
                if 'table_name' in table_detail:
                    ordered_detail['table_name'] = table_detail['table_name']
                
                if 'full_name' in table_detail:
                    ordered_detail['full_name'] = table_detail['full_name']
                
                # Add all other fields in their original order, excluding sample_data
                for key, value in table_detail.items():
                    if key not in ['schema', 'table_name', 'full_name', 'sample_data']:
                        ordered_detail[key] = value
                
                processed_matched_tables_details.append(ordered_detail)
            else:
                processed_matched_tables_details.append(table_detail)
        
        # Step 6: Create result JSON
        from datetime import datetime
        matched_tables_result = {
            "status": "success",
            "message": f"Successfully processed database {db_id} for user {user_id}",
            "metadata": {
                "user_id": user_id,
                "db_id": db_id,
                "db_name": db_config.get("db_name", ""),
                "generated_at": datetime.now().isoformat(),
                "total_business_rules_tables": len(business_rules_tables),
                "total_schema_tables": len(schema_tables),
                "total_matches": len(matched_tables)
            },
            "matched_tables": sorted(matched_tables),
            "matched_tables_details": processed_matched_tables_details,
            "business_rules_tables": sorted(business_rules_tables),
            "schema_tables": sorted(schema_tables),
            "unmatched_business_rules": _get_unmatched_tables(business_rules_tables, matched_tables),
            "unmatched_schema": _get_unmatched_tables(schema_tables, matched_tables)
        }
        
        # Step 7: Update the db_schema column with the generated data
        updated_config = db_manager.update_db_schema(db_id, matched_tables_result)
        
        if not updated_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update database schema"
            )
        
        # Step 8: Return success response with the generated data
        return APIResponse(
            status="success",
            message=f"Successfully generated matched tables for database {db_id} and updated schema",
            data={
                "db_id": db_id,
                "user_id": user_id,
                "matched_tables_data": matched_tables_result,
                "updated_config": updated_config
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating matched tables for db_id {db_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate matched tables: {str(e)}"
        )

@router.put("/mssql-config/{db_id}/report-structure", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["MSSQL Configuration"])
async def update_report_structure(db_id: int, update_data: ReportStructureUpdate):
    """
    Update the report structure for a specific database configuration
    
    - **db_id**: Database ID to update (from URL path)
    - **report_structure**: Report structure configuration as text to update (from request body)
    
    This endpoint allows updating only the report_structure column for a specific database configuration.
    The report_structure field is stored as TEXT and can contain any text data related to report configuration.
    You can store JSON strings, XML, or any other text format as needed.
    """
    try:
        # Check if database configuration exists
        existing_config = db_manager.get_mssql_config(db_id)
        if not existing_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MSSQL configuration with ID {db_id} not found"
            )
        
        # Update the report structure
        updated_config = db_manager.update_report_structure(db_id, update_data.report_structure)
        
        if updated_config:
            return APIResponse(
                status="success",
                message=f"Report structure updated successfully for database ID {db_id}",
                data={
                    "db_id": db_id,
                    "report_structure": update_data.report_structure,
                    "report_structure_length": len(update_data.report_structure),
                    "updated_config": updated_config
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update report structure"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating report structure for db_id {db_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update report structure: {str(e)}"
        )

# ============================================================================
# COMPANY MANAGEMENT ENDPOINTS
# ============================================================================

# ============================================================================
# MSSQL CONFIGURATION WORKFLOW ENDPOINTS
# ============================================================================

@router.post("/mssql-config/set-config", response_model=APIResponse, status_code=status.HTTP_202_ACCEPTED, tags=["MSSQL Configuration"])
async def start_set_config_workflow(
    db_url: Optional[str] = Form(None),
    db_name: str = Form(...),
    business_rule: str = Form(default=""),
    file: Optional[Union[UploadFile, str]] = File(None),
    user_id: str = Form(default="workflow"),
):
    """
    Start a background workflow to create a new config, generate table info/schema, and generate matched tables.
    Returns a task_id; check progress via GET /mssql-config/tasks/{task_id}.
    """
    try:
        # Read file bytes early (UploadFile stream may not be valid after response)
        file_bytes = await file.read() if (isinstance(file, UploadFile) and file.filename) else None
        original_filename = file.filename if (isinstance(file, UploadFile) and file.filename) else None

        # Create a task with placeholder db_id (-1) which will be updated later
        task_id = db_manager.create_table_info_task(user_id=user_id, db_id=-1)

        # Start background workflow
        asyncio.create_task(
            run_set_config_workflow(
                task_id=task_id,
                db_url=db_url or "",
                db_name=db_name,
                business_rule=business_rule,
                file_bytes=file_bytes,
                original_filename=original_filename,
            )
        )

        return APIResponse(
            status="accepted",
            message="Set-config workflow started",
            data={"task_id": task_id, "status": "pending"},
        )
    except Exception as e:
        logger.error(f"Error starting set-config workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start set-config workflow: {str(e)}",
        )


@router.put("/mssql-config/update-config/{db_id}", response_model=APIResponse, status_code=status.HTTP_202_ACCEPTED, tags=["MSSQL Configuration"])
async def start_update_config_workflow(
    db_id: int,
    db_url: str = Form(None),
    db_name: str = Form(None),
    business_rule: str = Form(None),
    file: Optional[Union[UploadFile, str]] = File(None),
    user_id: str = Form(default="workflow"),
):
    """
    Start a background workflow to update an existing config, generate table info/schema, and generate matched tables.
    Returns a task_id; check progress via GET /mssql-config/tasks/{task_id}.
    """
    try:
        # Validate exists
        config = db_manager.get_mssql_config(db_id)
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"MSSQL configuration with ID {db_id} not found",
            )

        # Build update model - only include non-empty fields
        update_fields: Dict[str, Any] = {}
        if db_url is not None and db_url.strip():
            update_fields["db_url"] = db_url.strip()
            logger.info(f"Update workflow: Including db_url field")
        if db_name is not None and db_name.strip():
            update_fields["db_name"] = db_name.strip()
            logger.info(f"Update workflow: Including db_name field")
        if business_rule is not None and business_rule.strip():
            update_fields["business_rule"] = business_rule.strip()
            logger.info(f"Update workflow: Including business_rule field")
        
        if not update_fields:
            logger.info(f"Update workflow: No non-empty fields to update, skipping database update")
        
        update_model = MSSQLDBConfigUpdate(**update_fields) if update_fields else MSSQLDBConfigUpdate()

        # Read file bytes early
        file_bytes = await file.read() if (isinstance(file, UploadFile) and file.filename) else None
        original_filename = file.filename if (isinstance(file, UploadFile) and file.filename) else None

        # Create task
        task_id = db_manager.create_table_info_task(user_id=user_id, db_id=db_id)

        # Start background workflow
        asyncio.create_task(
            run_update_config_workflow(
                task_id=task_id,
                db_id=db_id,
                update_model=update_model,
                file_bytes=file_bytes,
                original_filename=original_filename,
            )
        )

        return APIResponse(
            status="accepted",
            message=f"Update-config workflow started for db_id {db_id}",
            data={"task_id": task_id, "status": "pending", "db_id": db_id},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting update-config workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start update-config workflow: {str(e)}",
        )

@router.post("/parent-companies", response_model=APIResponse, status_code=status.HTTP_201_CREATED, tags=["Company Management"])
async def create_parent_company(company_data: ParentCompanyCreate):
    """
    Create a new parent company
    
    - **company_name**: Name of the parent company
    - **description**: Company description
    - **address**: Company address
    - **contact_email**: Contact email
    - **db_id**: Optional database ID to associate with parent company
    - **vector_db_id**: Optional vector database ID (must reference database_configs.db_id)
    """
    try:
        company = db_manager.create_parent_company(company_data)
        
        return APIResponse(
            status="success",
            message=f"Parent company '{company_data.company_name}' created successfully",
            data=company
        )
    except Exception as e:
        logger.error(f"Error creating parent company: {e}")
        raise e

@router.get("/parent-companies/{parent_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def get_parent_company(parent_company_id: int):
    """
    Get parent company by ID
    """
    try:
        company = db_manager.get_parent_company(parent_company_id)
        
        if company:
            return APIResponse(
                status="success",
                message="Parent company retrieved successfully",
                data=company
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent company with ID {parent_company_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving parent company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parent company: {str(e)}"
        )

@router.get("/parent-companies", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def get_all_parent_companies():
    """
    Get all parent companies
    """
    try:
        companies = db_manager.get_all_parent_companies()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(companies)} parent companies",
            data={
                "companies": companies,
                "count": len(companies)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving parent companies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parent companies: {str(e)}"
        )

@router.put("/parent-companies/{parent_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def update_parent_company(parent_company_id: int, update_data: ParentCompanyUpdate):
    """
    Update parent company
    
    - **company_name**: Updated company name (optional)
    - **description**: Updated description (optional)
    - **address**: Updated address (optional)
    - **contact_email**: Updated contact email (optional)
    - **db_id**: Updated database ID (optional)
    - **vector_db_id**: Updated vector database ID (must reference database_configs.db_id)
    """
    try:
        company = db_manager.update_parent_company(parent_company_id, update_data)
        
        if company:
            return APIResponse(
                status="success",
                message=f"Parent company ID {parent_company_id} updated successfully",
                data=company
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent company with ID {parent_company_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating parent company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update parent company: {str(e)}"
        )

@router.delete("/parent-companies/{parent_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def delete_parent_company(parent_company_id: int):
    """
    Delete parent company (will cascade to sub companies and user access)
    """
    try:
        # Check if company exists
        existing_company = db_manager.get_parent_company(parent_company_id)
        if not existing_company:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent company with ID {parent_company_id} not found"
            )
        
        db_manager.delete_parent_company(parent_company_id)
        
        return APIResponse(
            status="success",
            message=f"Parent company ID {parent_company_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting parent company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete parent company: {str(e)}"
        )

@router.post("/sub-companies", response_model=APIResponse, status_code=status.HTTP_201_CREATED, tags=["Company Management"])
async def create_sub_company(company_data: SubCompanyCreate):
    """
    Create a new sub company
    
    - **parent_company_id**: ID of the parent company
    - **company_name**: Name of the sub company
    - **description**: Company description
    - **address**: Company address
    - **contact_email**: Contact email
    - **db_id**: Optional database ID to associate with sub company
    - **vector_db_id**: Optional vector database ID (must reference database_configs.db_id)
    """
    try:
        company = db_manager.create_sub_company(company_data)
        
        return APIResponse(
            status="success",
            message=f"Sub company '{company_data.company_name}' created successfully",
            data=company
        )
    except Exception as e:
        logger.error(f"Error creating sub company: {e}")
        raise e

@router.get("/sub-companies/{sub_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def get_sub_company(sub_company_id: int):
    """
    Get sub company by ID
    """
    try:
        company = db_manager.get_sub_company(sub_company_id)
        
        if company:
            return APIResponse(
                status="success",
                message="Sub company retrieved successfully",
                data=company
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sub company with ID {sub_company_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving sub company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sub company: {str(e)}"
        )

@router.get("/sub-companies", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def get_all_sub_companies():
    """
    Get all sub companies with parent company information
    """
    try:
        companies = db_manager.get_all_sub_companies()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(companies)} sub companies",
            data={
                "companies": companies,
                "count": len(companies)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving sub companies: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sub companies: {str(e)}"
        )

@router.get("/parent-companies/{parent_company_id}/sub-companies", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def get_sub_companies_by_parent(parent_company_id: int):
    """
    Get all sub companies for a specific parent company
    """
    try:
        companies = db_manager.get_sub_companies_by_parent(parent_company_id)
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(companies)} sub companies for parent company ID {parent_company_id}",
            data={
                "parent_company_id": parent_company_id,
                "companies": companies,
                "count": len(companies)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving sub companies by parent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sub companies: {str(e)}"
        )

@router.put("/sub-companies/{sub_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def update_sub_company(sub_company_id: int, update_data: SubCompanyUpdate):
    """
    Update sub company
    
    - **company_name**: Updated company name (optional)
    - **description**: Updated description (optional)
    - **address**: Updated address (optional)
    - **contact_email**: Updated contact email (optional)
    - **db_id**: Updated database ID (optional)
    - **vector_db_id**: Updated vector database ID (must reference database_configs.db_id)
    """
    try:
        company = db_manager.update_sub_company(sub_company_id, update_data)
        
        if company:
            return APIResponse(
                status="success",
                message=f"Sub company ID {sub_company_id} updated successfully",
                data=company
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sub company with ID {sub_company_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating sub company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update sub company: {str(e)}"
        )

@router.delete("/sub-companies/{sub_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Company Management"])
async def delete_sub_company(sub_company_id: int):
    """
    Delete sub company
    """
    try:
        # Check if company exists
        existing_company = db_manager.get_sub_company(sub_company_id)
        if not existing_company:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sub company with ID {sub_company_id} not found"
            )
        
        db_manager.delete_sub_company(sub_company_id)
        
        return APIResponse(
            status="success",
            message=f"Sub company ID {sub_company_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting sub company: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete sub company: {str(e)}"
        )



# ============================================================================
# USER ACCESS CONTROL ENDPOINTS
# ============================================================================

@router.post("/user-access", response_model=APIResponse, status_code=status.HTTP_201_CREATED, tags=["User Access Control"])
async def create_user_access(access_config: UserAccessConfig):
    """
    Create or update user access configuration
    
    - **user_id**: Unique identifier for the user
    - **parent_company_id**: Parent company ID
    - **sub_company_ids**: List of sub company IDs this user can access
    - **database_access**: Specific database access control (optional)
    - **database_access_vector**: Specific database access vector control (optional)
    - **table_shows**: Tables to show for each database (JSON)
    """
    try:
        # Validate that all referenced databases exist if database_access is provided
        if access_config.database_access:
            # Validate parent company databases
            for db_access in access_config.database_access.parent_databases:
                if not db_manager.get_mssql_config(db_access.db_id):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Database with db_id {db_access.db_id} does not exist"
                    )
            
            # Validate sub company databases
            for sub_db_access in access_config.database_access.sub_databases:
                for db_access in sub_db_access.databases:
                    if not db_manager.get_mssql_config(db_access.db_id):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Database with db_id {db_access.db_id} does not exist"
                        )
        
        # Validate that all referenced databases exist if database_access_vector is provided
        if access_config.database_access_vector:
            # Validate parent company databases
            for db_access in access_config.database_access_vector.parent_databases:
                if not db_manager.get_mssql_config(db_access.db_id):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Database with db_id {db_access.db_id} does not exist in database_access_vector"
                    )
            
            # Validate sub company databases
            for sub_db_access in access_config.database_access_vector.sub_databases:
                for db_access in sub_db_access.databases:
                    if not db_manager.get_mssql_config(db_access.db_id):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Database with db_id {db_access.db_id} does not exist in database_access_vector"
                        )
        
        db_manager.insert_user_access(access_config)
        
        # Count total databases for response
        total_databases = 0
        if access_config.database_access:
            total_databases += len(access_config.database_access.parent_databases)
            for sub_db_access in access_config.database_access.sub_databases:
                total_databases += len(sub_db_access.databases)
        
        # Count total databases for vector response
        total_vector_databases = 0
        if access_config.database_access_vector:
            total_vector_databases += len(access_config.database_access_vector.parent_databases)
            for sub_db_access in access_config.database_access_vector.sub_databases:
                total_vector_databases += len(sub_db_access.databases)
        
        return APIResponse(
            status="success",
            message=f"User access configuration for '{access_config.user_id}' - parent company ID '{access_config.parent_company_id}' saved successfully",
            data={
                "user_id": access_config.user_id,
                "parent_company_id": access_config.parent_company_id,
                "sub_companies_count": len(access_config.sub_company_ids),
                "databases_count": total_databases,
                "vector_databases_count": total_vector_databases
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving user access config: {e}")
        raise e



@router.get("/user-access/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Access Control"])
async def get_user_access_by_user_id(user_id: str):
    """
    Retrieve all user access configurations for a specific user
    """
    try:
        access_configs = db_manager.get_user_access_by_user_id(user_id)
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(access_configs)} access configurations for user '{user_id}'",
            data={
                "user_id": user_id,
                "access_configs": access_configs,
                "count": len(access_configs)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving user access configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user access configurations: {str(e)}"
        )

@router.put("/user-access/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Access Control"])
async def update_user_access_by_user_id(user_id: str, update_data: UserAccessUpdate):
    """
    Update all user access configurations for a specific user
    
    - **sub_company_ids**: List of sub company IDs this user can access
    - **database_access**: Specific database access control (optional)
    - **database_access_vector**: Specific database access vector control (optional)
    - **table_shows**: Tables to show for each database (JSON)
    """
    try:
        updated_configs = db_manager.update_user_access_by_user_id(user_id, update_data)
        
        # Count total databases for response
        total_databases = 0
        if update_data.database_access:
            total_databases += len(update_data.database_access.parent_databases)
            for sub_db_access in update_data.database_access.sub_databases:
                total_databases += len(sub_db_access.databases)
        
        # Count total databases for vector response
        total_vector_databases = 0
        if update_data.database_access_vector:
            total_vector_databases += len(update_data.database_access_vector.parent_databases)
            for sub_db_access in update_data.database_access_vector.sub_databases:
                total_vector_databases += len(sub_db_access.databases)
        
        return APIResponse(
            status="success",
            message=f"Updated {len(updated_configs)} user access configurations for user '{user_id}'",
            data={
                "user_id": user_id,
                "updated_configs": updated_configs,
                "count": len(updated_configs),
                "sub_companies_count": len(update_data.sub_company_ids),
                "databases_count": total_databases,
                "vector_databases_count": total_vector_databases
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user access by user_id: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user access configuration: {str(e)}"
        )

@router.get("/user-access", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Access Control"])
async def get_all_user_access_configs():
    """
    Retrieve all user access configurations
    """
    try:
        access_configs = db_manager.get_all_user_access_configs()
        
        return APIResponse(
            status="success",
            message=f"Retrieved {len(access_configs)} user access configurations",
            data={
                "access_configs": access_configs,
                "count": len(access_configs)
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving all user access configs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user access configurations: {str(e)}"
        )

@router.delete("/user-access/{user_id}/{parent_company_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Access Control"])
async def delete_user_access(user_id: str, parent_company_id: int):
    """
    Delete user access configuration by user_id and parent_company_id
    """
    try:
        # Check if config exists
        existing_config = db_manager.get_user_access(user_id, parent_company_id)
        if not existing_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User access configuration not found for user_id: {user_id} and parent_company_id: {parent_company_id}"
            )
        
        db_manager.delete_user_access(user_id, parent_company_id)
        
        return APIResponse(
            status="success",
            message=f"User access configuration for '{user_id}' - parent company ID '{parent_company_id}' deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user access config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user access configuration: {str(e)}"
        )

# ============================================================================
# USER CURRENT DATABASE API ENDPOINTS
# ============================================================================

@router.put("/user-current-db/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Current Database"])
async def update_user_current_db(user_id: str, update_data: UserCurrentDBUpdate):
    """
    Update the current database for a user
    
    - **db_id**: Database ID to set as current for the user
    """
    try:
        result = db_manager.set_user_current_db(user_id, update_data.db_id)
        
        return APIResponse(
            status="success",
            message=f"Current database updated successfully for user '{user_id}'",
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user current database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user current database: {str(e)}"
        )

# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@router.delete("/cache/user/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Cache Management"])
async def clear_user_cache(user_id: str):
    """
    Clear all cached data for a specific user
    
    - **user_id**: User ID to clear cache for
    
    This will clear both full and lite database details from cache.
    """
    try:
        cleared_count = db_manager.clear_user_cache(user_id)
        
        return APIResponse(
            status="success",
            message=f"Cache cleared for user '{user_id}'",
            data={
                "user_id": user_id,
                "cleared_entries": cleared_count
            }
        )
    except Exception as e:
        logger.error(f"Error clearing cache for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache for user: {str(e)}"
        )

@router.delete("/cache/all", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Cache Management"])
async def clear_all_cache():
    """
    Clear all cached data for all users
    
    This will completely clear the cache, forcing all subsequent requests to fetch fresh data.
    """
    try:
        cleared_count = user_db_cache.clear()
        
        return APIResponse(
            status="success",
            message="All cache cleared successfully",
            data={
                "cleared_entries": cleared_count,
                "cache_stats": user_db_cache.get_stats()
            }
        )
    except Exception as e:
        logger.error(f"Error clearing all cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear all cache: {str(e)}"
        )

@router.get("/cache/stats", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["Cache Management"])
async def get_cache_stats():
    """
    Get cache statistics and performance metrics
    
    Returns information about cache hit rates, size, and other performance metrics.
    """
    try:
        stats = user_db_cache.get_stats()
        cache_keys = user_db_cache.get_all_keys()
        
        return APIResponse(
            status="success",
            message="Cache statistics retrieved successfully",
            data={
                "stats": stats,
                "active_keys": cache_keys,
                "active_keys_count": len(cache_keys)
            }
        )
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache statistics: {str(e)}"
        )



@router.get("/user-current-db/{user_id}", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Current Database"])
async def get_user_current_db_details(user_id: str):
    """
    Get the current database details for a user including business_rule, table_info, db_schema, report_structure, db_id
    
    This is the comprehensive endpoint that returns all database details including large fields.
    Automatic Redis caching enabled for optimal performance.
    
    - **user_id**: User ID to get database details for
    
    **Performance Features:**
    - ⚡ Automatic Redis caching (10s TTL)
    - 💾 Persistent cache with Redis backend
    - 🔄 Automatic cache invalidation on data updates
    - ✨ Up to 90% faster response times for cached requests
    
    **For better performance consider:**
    - Use `/lite` endpoint for UI components (80% smaller payload)
    - Use `/selective` endpoint for custom field selection
    """
    try:
        result = db_manager.get_user_current_db_details(user_id)
        
        if result:
            return APIResponse(
                status="success",
                message=f"Current database details retrieved successfully for user '{user_id}' with automatic Redis caching",
                data=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user current database details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user current database details: {str(e)}"
        )

@router.get("/user-current-db/{user_id}/lite", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Current Database"])
async def get_user_current_db_details_lite(user_id: str):
    """    
    Get the current database details for a user with selective field loading (excludes table_info only)
    
    This endpoint provides faster responses by excluding large fields like table_info.
    Now includes db_schema as requested while maintaining performance optimization.
    Ideal for UI components that need database schema information but not table_info.
    Automatic Redis caching enabled for optimal performance.
    
    - **user_id**: User ID to get database details for
    
    **Excluded fields for better performance:**
    - table_info (large JSON object)
    
    **Included fields:**
    - user_id, db_id, db_name, db_url
    - business_rule, db_schema, report_structure
    - created_at, updated_at
    - has_table_info, has_db_schema (boolean indicators)
    
    **Performance Features:**
    - ⚡ Automatic Redis caching (10s TTL)
    - 📊 Optimized payload (excludes only table_info)
    - 🚀 Optimized for dashboards and UI components needing schema info
    """
    try:
        result = db_manager.get_user_current_db_details_lite(user_id)
        
        if result:
            return APIResponse(
                status="success",
                message=f"Current database details (lite) retrieved successfully for user '{user_id}' with automatic caching",
                data=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user current database details (lite): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user current database details (lite): {str(e)}"
        )

@router.get("/user-current-db/{user_id}/selective", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["User Current Database"])
async def get_user_current_db_details_selective(
    user_id: str,
    include_table_info: bool = Query(False, description="Include table_info field (large JSON)"),
    include_db_schema: bool = Query(False, description="Include db_schema field (very large JSON)")
):
    """
    Get the current database details for a user with selective field loading
    
    This endpoint allows fine-grained control over which large fields to include in the response.
    Use this when you need specific large fields but want to optimize performance.
    Automatic Redis caching with smart cache keys based on field selection.
    
    - **user_id**: User ID to get database details for
    - **include_table_info**: Include table_info field (default: False)
    - **include_db_schema**: Include db_schema field (default: False)
    
    **Performance Features:**
    - ⚡ Smart Redis caching (different cache keys per field combination)
    - 📊 Configurable payload size optimization
    - 🚀 Optimal for custom API integrations
    
    **Performance Note:** Including both large fields may result in slower response times.
    """
    try:
        # Use the updated method with automatic caching
        result = db_manager.get_user_current_db_details_selective(user_id, include_table_info, include_db_schema)
        
        if result:
            # Extract metadata if present
            metadata = result.get('_metadata', {})
            
            fields_included = []
            if include_table_info:
                fields_included.append('table_info')
            if include_db_schema:
                fields_included.append('db_schema')
            
            fields_msg = f" (included: {', '.join(fields_included)})" if fields_included else " (basic fields only)"
            
            return APIResponse(
                status="success",
                message=f"Current database details retrieved successfully for user '{user_id}' with automatic caching{fields_msg}",
                data=result
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No current database found for user '{user_id}'"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user current database details (selective): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user current database details (selective): {str(e)}"
        )

@router.get("/health", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["System"])
async def mssql_config_health_check():
    """
    Health check endpoint
    """
    try:
        # Test database connection
        conn = db_manager.get_connection()
        conn.close()
        
        # Check if database existed before setup
        db_existed = db_manager.database_exists(db_manager.target_db)
        
        return APIResponse(
            status="healthy",
            message="MSSQL Configuration Manager API and database are running",
            data={
                "database": db_manager.target_db,
                "database_existed_before_setup": db_existed,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )

@router.get("/database-status", response_model=APIResponse, status_code=status.HTTP_200_OK, tags=["System"])
async def database_status():
    """
    Get detailed database status and safety information
    """
    try:
        target_db = db_manager.target_db
        db_exists = db_manager.database_exists(target_db)
        
        # Check if tables exist (if database exists)
        table_status = {}
        if db_exists:
            try:
                conn = db_manager.get_connection()
                with conn.cursor() as cursor:
                    # Check if tables exist
                    tables_to_check = ['mssql_config', 'user_access']
                    for table in tables_to_check:
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_schema = 'public' 
                                AND table_name = %s
                            );
                        """, (table,))
                        exists = cursor.fetchone()[0]
                        table_status[table] = exists
                conn.close()
            except Exception as e:
                logger.warning(f"Could not check table status: {e}")
        
        return APIResponse(
            status="success",
            message="Database status retrieved successfully",
            data={
                "target_database": target_db,
                "database_exists": db_exists,
                "table_status": table_status,
                "safety_info": {
                    "database_creation_safe": True,
                    "table_creation_safe": True,
                    "no_data_overwrite": True
                },
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error getting database status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database status: {str(e)}"
        )

app.include_router(router)



# if __name__ == '__main__':
#     print("Starting MSSQL Configuration Manager API v2.0 with FastAPI...")
#     print("Available endpoints by category:")
#     print("\n📋 System:")
#     print("  POST /setup - Setup database and tables")
#     print("  GET /health - Health check")
#     print("  GET /database-status - Get detailed database status and safety info")
#     print("\n🗄️ MSSQL Configuration:")
#     print("  POST /mssql-config - Create new MSSQL configuration (auto-generated db_id)")
#     print("  GET /mssql-config/{db_id} - Get specific MSSQL configuration")
#     print("  PUT /mssql-config/{db_id} - Update MSSQL configuration")
#     print("  GET /mssql-config - Get all MSSQL configurations")
#     print("  DELETE /mssql-config/{db_id} - Delete MSSQL configuration")
#     print("  POST /mssql-config/{db_id}/upload-file - Upload database recovery file")
#     print("  POST /mssql-config/{db_id}/generate-table-info - Start background table info generation")
#     print("  GET /mssql-config/tasks/{task_id} - Get table info generation task status")
#     print("\n🏢 Company Management:")
#     print("  POST /parent-companies - Create parent company")
#     print("  GET /parent-companies/{id} - Get parent company by ID")
#     print("  GET /parent-companies - Get all parent companies")
#     print("  PUT /parent-companies/{id} - Update parent company")
#     print("  DELETE /parent-companies/{id} - Delete parent company")
#     print("  POST /sub-companies - Create sub company")
#     print("  GET /sub-companies/{id} - Get sub company by ID")
#     print("  GET /sub-companies - Get all sub companies")
#     print("  GET /parent-companies/{id}/sub-companies - Get sub companies by parent")
#     print("  PUT /sub-companies/{id} - Update sub company")
#     print("  DELETE /sub-companies/{id} - Delete sub company")

#     print("\n👤 User Access Control:")
#     print("  POST /user-access - Create/update user access configuration")
#     print("  GET /user-access/{user_id} - Get all access configurations for a user")
#     print("  PUT /user-access/{user_id} - Update all access configurations for a user")
#     print("  GET /user-access - Get all user access configurations")
#     print("  DELETE /user-access/{user_id}/{parent_company_id} - Delete user access")
#     print("\nAPI Documentation:")
#     print("  Swagger UI: http://localhost:8000/docs")
#     print("  ReDoc: http://localhost:8000/redoc")
#     print("\nStarting server on http://localhost:8000")
    
#     uvicorn.run(
#         "mssql_config:app", 
#         host="0.0.0.0", 
#         port=8000, 
#         reload=True,
#         log_level="info"
#     )
