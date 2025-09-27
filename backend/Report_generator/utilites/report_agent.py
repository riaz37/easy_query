import os
import sys
import json
import pandas as pd
import requests
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from io import BytesIO
from dotenv import load_dotenv
import time
from datetime import datetime, date, timedelta
import threading
import copy

# Environment Configuration
# Set BASE_URL environment variable to override the default API URL
# Example: export BASE_URL=https://127.0.0.1:8200
# Default: http://localhost:8000

# Plotly imports - Much more beautiful and modern than Bokeh
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
import kaleido  # For image export

# FastAPI imports
from fastapi import APIRouter, HTTPException, Body, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# LLM imports
from langchain_google_genai import ChatGoogleGenerativeAI

# Database imports with performance optimizations
import psycopg2
from psycopg2.extras import RealDictCursor
import sqlite3
from contextlib import contextmanager
from enum import Enum
import uuid

# Performance optimization imports
from concurrent.futures import ThreadPoolExecutor
import functools
import weakref

# Add the project root to the path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(BASE_DIR)

# Import the query function directly from mssql_agent3
try:
    from data_sources.mssql.mssql_agent3 import query_database
    DIRECT_QUERY_AVAILABLE = True
except ImportError:
    print("Warning: Could not import query_database directly, will use HTTP fallback")
    DIRECT_QUERY_AVAILABLE = False

# Import graph generation functions directly from graph_Generator.py
try:
    from Report_generator.utilites.graph_Generator import generate_beautiful_graph, generate_multiple_graphs, get_server_base_url, get_image_server_url
    DIRECT_GRAPH_AVAILABLE = True
    print("✅ Successfully imported direct graph generation functions")
except ImportError:
    print("Warning: Could not import graph generation functions directly, will use HTTP fallback")
    DIRECT_GRAPH_AVAILABLE = False

# Import database manager for direct function calls
try:
    from db_manager.mssql_config import db_manager
    DIRECT_DB_AVAILABLE = True
    print("✅ Successfully imported database manager for direct function calls")
except ImportError:
    print("Warning: Could not import database manager, will use HTTP fallback")
    DIRECT_DB_AVAILABLE = False

# Initialize environment and LLM
load_dotenv(override=True)
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")

# Performance optimization: Connection pooling
class DatabaseConnectionPool:
    """Optimized database connection pool for better performance"""

    def __init__(self, max_connections=10):
        self.max_connections = max_connections
        self.connections = []
        self.available = []
        self.lock = threading.Lock()
        self._cleanup_thread = None
        self._running = True

    def get_connection(self):
        """Get a database connection from the pool"""
        with self.lock:
            if self.available:
                conn = self.available.pop()
                try:
                    # Test connection
                    conn.cursor().execute("SELECT 1")
                    return conn
                except:
                    # Connection is bad, create new one
                    pass

            if len(self.connections) < self.max_connections:
                conn = self._create_connection()
                if conn:
                    self.connections.append(conn)
                    return conn

        # If we can't get a connection, create a temporary one
        return self._create_connection()

    def return_connection(self, conn):
        """Return a connection to the pool"""
        with self.lock:
            if conn and conn in self.connections:
                try:
                    self.available.append(conn)
                except:
                    pass

    def _create_connection(self):
        """Create a new database connection"""
        try:
            conn = psycopg2.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                connect_timeout=30
            )
            return conn
        except Exception as e:
            print(f"❌ Error creating database connection: {e}")
            return None

    def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
            self.available.clear()

# Initialize connection pool
db_pool = DatabaseConnectionPool(max_connections=5)

# Performance optimization: Thread pool for background tasks
background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ReportTask")

# Performance optimization: LRU Cache for frequently accessed data
@functools.lru_cache(maxsize=100)
def cached_get_db_id_from_user_id_sync(user_id: str):
    """Cached version of database ID lookup"""
    return get_db_id_from_user_id_sync(user_id)

@functools.lru_cache(maxsize=50)
def cached_get_report_structure_from_db(db_id: int):
    """Cached version of report structure lookup"""
    return get_report_structure_from_db(db_id)

# Pydantic Models for API
class ReportGenerationRequest(BaseModel):
    """Request model for report generation API"""
    user_id: str
    user_query: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "user_id": "nabil",
                "user_query": "financial report 2023"
            }
        }

class ReportGenerationResponse(BaseModel):
    """Response model for report generation API"""
    success: bool
    user_id: str
    timestamp: str
    generation_time: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    error: Optional[str] = None
    database_id: Optional[Union[str, int]] = None

# Task Management Models
class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ReportTaskInfo(BaseModel):
    """Report task information model"""
    task_id: str
    user_id: str
    user_query: Optional[str] = None
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: str = "Pending"
    current_step: Optional[str] = None
    current_query: Optional[str] = None
    total_queries: int = 0
    processed_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    database_id: Optional[int] = None

class BackgroundReportResponse(BaseModel):
    """Response model for background report generation"""
    task_id: str
    status: str
    message: str
    user_id: str
    user_query: Optional[str] = None
    timestamp: str

class ReportTaskStatusResponse(BaseModel):
    """Response model for report task status"""
    task_id: str
    user_id: str
    user_query: Optional[str] = None
    status: str
    progress: str
    current_step: Optional[str] = None
    current_query: Optional[str] = None
    total_queries: int
    processed_queries: int
    successful_queries: int
    failed_queries: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    progress_percentage: float
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    database_id: Optional[int] = None

    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    progress_percentage: float
    error: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    database_id: Optional[int] = None

# Report Task Manager for Background Processing with performance optimizations
class ReportTaskManager:
    """Database manager for report background tasks with performance optimizations"""

    def __init__(self, db_path: str = "report_task_tracking.db"):
        self.db_path = db_path
        self._local = threading.local()  # Thread-local storage for connections
        self._connection_cache = {}  # Cache connections per thread
        self._cache_lock = threading.Lock()
        self.init_database()

    def init_database(self):
        """Initialize the SQLite database with required tables for report tasks."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")

            # Create report_tasks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS report_tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_query TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress TEXT DEFAULT 'Pending',
                    current_step TEXT,
                    total_queries INTEGER DEFAULT 0,
                    processed_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    failed_queries INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    error TEXT,
                    results TEXT,  -- JSON string
                    database_id INTEGER
                )
            ''')

            # Create optimized indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_report_tasks_user_id ON report_tasks(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_report_tasks_status ON report_tasks(status)",
                "CREATE INDEX IF NOT EXISTS idx_report_tasks_created_at ON report_tasks(created_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_report_tasks_user_status ON report_tasks(user_id, status)"
            ]

            for index_sql in indexes:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e):
                        print(f"⚠️  Could not create index: {e}")

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections with connection pooling"""
        thread_id = threading.get_ident()

        with self._cache_lock:
            if thread_id not in self._connection_cache:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                # Enable optimizations
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                self._connection_cache[thread_id] = conn

        conn = self._connection_cache[thread_id]

        try:
            # Test connection
            conn.execute("SELECT 1").fetchone()
            yield conn
        except:
            # Connection is bad, recreate it
            try:
                conn.close()
            except:
                pass

            with self._cache_lock:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA cache_size=10000")
                self._connection_cache[thread_id] = conn

            yield conn

    def create_task(self, task_info: ReportTaskInfo):
        """Create a new report task in the database."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO report_tasks (
                    task_id, user_id, user_query, status, created_at, started_at, completed_at,
                    progress, current_step, current_query, total_queries, processed_queries, successful_queries,
                    failed_queries, processing_time_seconds, error, results, database_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_info.task_id,
                task_info.user_id,
                task_info.user_query,
                task_info.status.value,
                task_info.created_at.isoformat(),
                task_info.started_at.isoformat() if task_info.started_at else None,
                task_info.completed_at.isoformat() if task_info.completed_at else None,
                task_info.progress,
                task_info.current_step,
                task_info.current_query,
                task_info.total_queries,
                task_info.processed_queries,
                task_info.successful_queries,
                task_info.failed_queries,
                task_info.processing_time_seconds,
                task_info.error,
                json.dumps(task_info.results) if task_info.results else None,
                task_info.database_id
            ))
            conn.commit()

    def get_task(self, task_id: str) -> Optional[ReportTaskInfo]:
        """Get a report task by ID."""
        with self.get_connection() as conn:
            row = conn.execute('SELECT * FROM report_tasks WHERE task_id = ?', (task_id,)).fetchone()
            if row:
                return self._row_to_task_info(row)
            return None

    def update_task(self, task_id: str, **kwargs):
        """Update report task fields."""
        if not kwargs:
            return

        # Convert datetime objects to ISO format and enum values
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                kwargs[key] = value.isoformat()
            elif isinstance(value, TaskStatus):
                kwargs[key] = value.value
            elif key == 'results' and value is not None:
                kwargs[key] = json.dumps(value)

        set_clause = ', '.join([f"{key} = ?" for key in kwargs.keys()])
        query = f"UPDATE report_tasks SET {set_clause} WHERE task_id = ?"

        with self.get_connection() as conn:
            conn.execute(query, list(kwargs.values()) + [task_id])
            conn.commit()

    def _row_to_task_info(self, row) -> ReportTaskInfo:
        """Convert database row to ReportTaskInfo object."""
        return ReportTaskInfo(
            task_id=row['task_id'],
            user_id=row['user_id'],
            user_query=row['user_query'],
            status=TaskStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            progress=row['progress'],
            current_step=row['current_step'],
            current_query=row['current_query'],
            total_queries=row['total_queries'],
            processed_queries=row['processed_queries'],
            successful_queries=row['successful_queries'],
            failed_queries=row['failed_queries'],
            processing_time_seconds=row['processing_time_seconds'],
            error=row['error'],
            results=json.loads(row['results']) if row['results'] else None,
            database_id=row['database_id']
        )

    def get_user_tasks(self, user_id: str, limit: int = 50) -> List[ReportTaskInfo]:
        """Get all tasks for a user."""
        with self.get_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM report_tasks WHERE user_id = ? ORDER BY created_at DESC LIMIT ?',
                (user_id, limit)
            ).fetchall()
            return [self._row_to_task_info(row) for row in rows]

# Initialize the report task manager
report_task_manager = ReportTaskManager()

def recreate_report_task_database():
    """Recreate the report task database with the correct schema."""
    import os

    db_path = "report_task_tracking.db"

    # Remove existing database file
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"🗑️  Removed existing database: {db_path}")
        except Exception as e:
            print(f"⚠️  Could not remove existing database: {e}")

    # Create new database with correct schema
    try:
        new_manager = ReportTaskManager(db_path)
        print(f"✅ Created new database with correct schema: {db_path}")
        return new_manager
    except Exception as e:
        print(f"❌ Error creating new database: {e}")
        return None

# Initialize FastAPI Router
router = APIRouter()

# Performance optimized background task function
async def run_background_report_generation(task_id: str, user_id: str, user_query: Optional[str] = None):
    """
    Optimized background task function that runs report generation with comprehensive tracking.

    Args:
        task_id (str): Unique task identifier
        user_id (str): User ID for report generation
        user_query (str, optional): User's specific query for customization
    """

    def run_in_thread():
        """Run the report generation in a separate thread for better performance"""
        try:
            print(f"🚀 Starting background report generation for task {task_id}")

            # Update task status to running
            task_info = report_task_manager.get_task(task_id)
            if not task_info:
                print(f"❌ Task {task_id} not found in database")
                return

            # Mark task as running
            report_task_manager.update_task(
                task_id,
                status=TaskStatus.RUNNING,
                started_at=datetime.now(),
                progress="Initializing report generation",
                current_step="Getting user database ID"
            )

            start_time = time.time()

            # Step 1: Get database ID for user
            print(f"📊 Getting database ID for user: {user_id}")
            report_task_manager.update_task(
                task_id,
                progress="Fetching user database information",
                current_step="Database ID lookup"
            )

            try:
                if DIRECT_DB_AVAILABLE:
                    user_data = db_manager.get_user_current_db_details(user_id)
                    if not user_data or not user_data.get('db_id'):
                        raise Exception(f"No current database found for user {user_id}")
                    db_id = user_data['db_id']
                else:
                    db_id = cached_get_db_id_from_user_id_sync(user_id)  # Use cached version

                print(f"✅ Found database ID {db_id} for user {user_id}")

                # Update task with database ID
                report_task_manager.update_task(
                    task_id,
                    database_id=db_id,
                    progress="Database ID retrieved successfully",
                    current_step="Parsing report structure"
                )

            except Exception as e:
                error_msg = f"Failed to get database ID: {str(e)}"
                print(f"❌ {error_msg}")
                report_task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    completed_at=datetime.now(),
                    processing_time_seconds=time.time() - start_time
                )
                return

            # Step 2: Parse report structure and get queries
            print(f"📖 Parsing report structure for database ID {db_id}")
            report_task_manager.update_task(
                task_id,
                progress="Parsing report structure from database",
                current_step="Query extraction"
            )

            try:
                # Use cached version for better performance
                report_structure_content = cached_get_report_structure_from_db(db_id)
                sections_data = parse_report_structure_with_llm(report_structure_content, user_query)

                if not sections_data or "sections" not in sections_data:
                    raise Exception("No report structure found or failed to parse")

                # Count total queries
                total_queries = 0
                for section in sections_data["sections"]:
                    total_queries += len(section.get("queries", []))

                print(f"📊 Found {total_queries} queries to process")

                # Update task with query count
                report_task_manager.update_task(
                    task_id,
                    total_queries=total_queries,
                    progress=f"Found {total_queries} queries to process",
                    current_step="Starting query processing"
                )

            except Exception as e:
                error_msg = f"Failed to parse report structure: {str(e)}"
                print(f"❌ {error_msg}")
                report_task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    error=error_msg,
                    completed_at=datetime.now(),
                    processing_time_seconds=time.time() - start_time
                )
                return

            # Step 3: Process queries with progress tracking
            print(f"🔄 Starting optimized query processing")
            report_task_manager.update_task(
                task_id,
                progress="Starting optimized batch processing",
                current_step="Query execution and graph generation"
            )

            try:
                # Use the optimized processing function with task_id for current query tracking
                result = process_all_queries_with_graph_generation_optimized_sync(
                    db_id=db_id,
                    user_id=user_id,
                    export_format="png",
                    theme="modern",
                    analysis_subject="data analysis",
                    user_query=user_query,
                    task_id=task_id  # Pass task_id for current query tracking
                )

                total_time = time.time() - start_time

                # Step 4: Finalize task based on result
                if result.get("success"):
                    print(f"🎉 Report generation completed successfully!")

                    # Serialize results for storage
                    serialized_result = serialize_for_json(result)

                    # Update task as completed
                    report_task_manager.update_task(
                        task_id,
                        status=TaskStatus.COMPLETED,
                        completed_at=datetime.now(),
                        processing_time_seconds=total_time,
                        progress=f"Report generation completed successfully",
                        current_step="Completed",
                        processed_queries=result.get("total_queries", 0),
                        successful_queries=result.get("successful_queries", 0),
                        failed_queries=result.get("failed_queries", 0),
                        results=serialized_result
                    )

                    print(f"✅ Task {task_id} completed in {total_time:.2f} seconds")
                    print(f"📊 Success rate: {result.get('summary', {}).get('success_rate', 0):.1f}%")

                else:
                    error_msg = result.get("error", "Unknown error in report generation")
                    print(f"❌ Report generation failed: {error_msg}")

                    report_task_manager.update_task(
                        task_id,
                        status=TaskStatus.FAILED,
                        completed_at=datetime.now(),
                        processing_time_seconds=total_time,
                        error=error_msg,
                        progress="Report generation failed",
                        current_step="Failed"
                    )

            except Exception as e:
                error_msg = f"Error during query processing: {str(e)}"
                print(f"❌ {error_msg}")

                report_task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    completed_at=datetime.now(),
                    processing_time_seconds=time.time() - start_time,
                    error=error_msg,
                    progress="Query processing failed",
                    current_step="Failed"
                )

        except Exception as e:
            error_msg = f"Critical error in background report generation: {str(e)}"
            print(f"💥 {error_msg}")
            import traceback
            traceback.print_exc()

            # Update task as failed
            report_task_manager.update_task(
                task_id,
                status=TaskStatus.FAILED,
                completed_at=datetime.now(),
                processing_time_seconds=time.time() - start_time if 'start_time' in locals() else 0,
                error=error_msg,
                progress="Critical failure in report generation",
                current_step="Failed"
            )

    # Run in thread pool for better performance
    future = background_executor.submit(run_in_thread)
    return future

@router.post("/generate-report-background", response_model=BackgroundReportResponse)
async def generate_report_background(
    background_tasks: BackgroundTasks,
    request: ReportGenerationRequest
):
    """
    Start background report generation and return task ID immediately.

    This endpoint:
    1. Creates a background task for report generation
    2. Returns task ID immediately for status tracking
    3. Allows frontend to poll for progress updates
    4. Suitable for long-running report generation processes

    Args:
        background_tasks: FastAPI BackgroundTasks dependency
        request: ReportGenerationRequest containing user_id and optional user_query

    Returns:
        BackgroundReportResponse with task_id for status tracking
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        timestamp = datetime.now()

        print(f"🚀 Creating background report generation task {task_id}")
        print(f"👤 User ID: {request.user_id}")
        if request.user_query:
            print(f"🎯 User Query: {request.user_query}")

        # Create task info
        task_info = ReportTaskInfo(
            task_id=task_id,
            user_id=request.user_id,
            user_query=request.user_query,
            status=TaskStatus.PENDING,
            created_at=timestamp,
            progress="Task created and queued for processing"
        )

        # Store task in database
        report_task_manager.create_task(task_info)

        # Submit to optimized thread pool
        future = background_executor.submit(
            lambda: asyncio.run(run_background_report_generation(task_id, request.user_id, request.user_query))
        )

        print(f"✅ Background task {task_id} created and submitted to optimized thread pool")

        return BackgroundReportResponse(
            task_id=task_id,
            status="accepted",
            message=f"Report generation started in background. Use task_id to check progress.",
            user_id=request.user_id,
            user_query=request.user_query,
            timestamp=timestamp.isoformat()
        )

    except Exception as e:
        print(f"❌ Error creating background task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create background report generation task: {str(e)}"
        )

@router.get("/task-status/{task_id}", response_model=ReportTaskStatusResponse)
async def get_report_task_status(task_id: str):
    """
    Get the status and progress of a background report generation task.

    This endpoint provides real-time status updates for frontend polling:
    1. Task status (pending, running, completed, failed)
    2. Progress percentage and current step
    3. Query processing statistics
    4. Results when completed
    5. Error details if failed

    Args:
        task_id: Unique task identifier from generate-report-background

    Returns:
        ReportTaskStatusResponse with comprehensive task information
    """
    try:
        # Get task from database
        task_info = report_task_manager.get_task(task_id)

        if not task_info:
            raise HTTPException(
                status_code=404,
                detail=f"Task with ID {task_id} not found"
            )

        # Calculate progress percentage
        progress_percentage = 0.0
        if task_info.total_queries > 0:
            progress_percentage = (task_info.processed_queries / task_info.total_queries) * 100
        elif task_info.status == TaskStatus.COMPLETED:
            progress_percentage = 100.0
        elif task_info.status == TaskStatus.RUNNING:
            progress_percentage = 10.0  # Minimum progress for running tasks

        # Create response
        response = ReportTaskStatusResponse(
            task_id=task_info.task_id,
            user_id=task_info.user_id,
            user_query=task_info.user_query,
            status=task_info.status.value,
            progress=task_info.progress,
            current_step=task_info.current_step,
            total_queries=task_info.total_queries,
            processed_queries=task_info.processed_queries,
            successful_queries=task_info.successful_queries,
            failed_queries=task_info.failed_queries,
            created_at=task_info.created_at.isoformat(),
            started_at=task_info.started_at.isoformat() if task_info.started_at else None,
            completed_at=task_info.completed_at.isoformat() if task_info.completed_at else None,
            processing_time_seconds=task_info.processing_time_seconds,
            progress_percentage=progress_percentage,
            error=task_info.error,
            results=task_info.results,
            database_id=task_info.database_id
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting task status for {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

@router.get("/user-tasks/{user_id}")
async def get_user_report_tasks(user_id: str, limit: int = 20):
    """
    Get all report generation tasks for a specific user.

    Useful for frontend to show task history and allow users to:
    1. See their recent report generation requests
    2. Check status of multiple tasks
    3. Access completed report results

    Args:
        user_id: User ID to get tasks for
        limit: Maximum number of tasks to return (default: 20)

    Returns:
        List of ReportTaskStatusResponse objects
    """
    try:
        # Get tasks from database
        tasks = report_task_manager.get_user_tasks(user_id, limit)

        # Convert to response format
        response_tasks = []
        for task_info in tasks:
            # Calculate progress percentage
            progress_percentage = 0.0
            if task_info.total_queries > 0:
                progress_percentage = (task_info.processed_queries / task_info.total_queries) * 100
            elif task_info.status == TaskStatus.COMPLETED:
                progress_percentage = 100.0
            elif task_info.status == TaskStatus.RUNNING:
                progress_percentage = 10.0

            task_response = ReportTaskStatusResponse(
                task_id=task_info.task_id,
                user_id=task_info.user_id,
                user_query=task_info.user_query,
                status=task_info.status.value,
                progress=task_info.progress,
                current_step=task_info.current_step,
                total_queries=task_info.total_queries,
                processed_queries=task_info.processed_queries,
                successful_queries=task_info.successful_queries,
                failed_queries=task_info.failed_queries,
                created_at=task_info.created_at.isoformat(),
                started_at=task_info.started_at.isoformat() if task_info.started_at else None,
                completed_at=task_info.completed_at.isoformat() if task_info.completed_at else None,
                processing_time_seconds=task_info.processing_time_seconds,
                progress_percentage=progress_percentage,
                error=task_info.error,
                results=task_info.results,  # Include results for completed tasks
                database_id=task_info.database_id
            )
            response_tasks.append(task_response)

        return {
            "user_id": user_id,
            "total_tasks": len(response_tasks),
            "tasks": response_tasks
        }

    except Exception as e:
        print(f"❌ Error getting user tasks for {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user tasks: {str(e)}"
        )

@router.post("/recreate-database")
async def recreate_database():
    """
    Recreate the report task database with the correct schema.
    Use this endpoint if you encounter database schema issues.

    Returns:
        Success message
    """
    try:
        global report_task_manager

        print("🔄 Recreating report task database...")

        # Recreate the database
        new_manager = recreate_report_task_database()

        if new_manager:
            # Update the global manager
            report_task_manager = new_manager
            print("✅ Database recreated successfully")

            return {
                "success": True,
                "message": "Report task database recreated successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to recreate database")

    except Exception as e:
        print(f"❌ Error recreating database: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to recreate database: {str(e)}"
        )

@router.get("/test-url-conversion")
async def test_url_conversion():
    """
    Test endpoint to verify URL conversion is working correctly.
    This endpoint tests the convert_file_path_to_url function directly.

    Returns:
        Test result with URL conversion details
    """
    try:
        print("🧪 Testing URL conversion...")

        # Import the conversion function
        from Report_generator.utilites.graph_Generator import convert_file_path_to_url, get_image_server_url, get_server_base_url

        # Test data
        test_image_path = "/path/to/storage/graphs/images/test_graph_123456.png"
        test_html_path = "/path/to/storage/graphs/html/test_graph_123456.html"
        test_other_path = "/path/to/other/file.txt"

        # Test conversions
        image_url = convert_file_path_to_url(test_image_path)
        html_url = convert_file_path_to_url(test_html_path)
        other_url = convert_file_path_to_url(test_other_path)

        # Get server URLs for comparison
        remote_url = get_image_server_url()
        local_url = get_server_base_url()

        return {
            "success": True,
            "message": "URL conversion test completed",
            "test_results": {
                "image_path": test_image_path,
                "image_url": image_url,
                "html_path": test_html_path,
                "html_url": html_url,
                "other_path": test_other_path,
                "other_url": other_url,
                "server_urls": {
                    "remote_image_server": remote_url,
                    "local_server": local_url
                },
                "environment": {
                    "SERVER_URL": os.getenv("SERVER_URL"),
                    "BASE_URL": os.getenv("BASE_URL")
                }
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ Error testing URL conversion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test URL conversion: {str(e)}"
        )

@router.get("/test-image-urls")
async def test_image_urls():
    """
    Test endpoint to verify that image URLs are working correctly.
    This endpoint generates a simple test graph and returns accessible URLs.

    Returns:
        Test result with image URLs
    """
    try:
        print("🧪 Testing image URL generation...")

        # Check environment variables first
        server_url = os.getenv("SERVER_URL")
        base_url = os.getenv("BASE_URL")

        print(f"🔍 Environment check:")
        print(f"   SERVER_URL: {server_url}")
        print(f"   BASE_URL: {base_url}")

        # Test URL generation functions
        try:
            remote_url = get_image_server_url()
            local_url = get_server_base_url()
            print(f"✅ URL functions working:")
            print(f"   Remote image URL: {remote_url}")
            print(f"   Local server URL: {local_url}")
        except Exception as e:
            print(f"❌ URL functions failed: {e}")

        # Import the graph generation function
        from Report_generator.utilites.graph_Generator import generate_beautiful_graph

        # Generate a simple test graph
        test_result = await generate_beautiful_graph(
            query="SELECT TOP 5 * FROM employee_salary",
            user_id="test_user",
            export_format="png",
            theme="modern"
        )

        if test_result.get("success"):
            image_url = test_result.get("files", {}).get("image_url", "")
            html_url = test_result.get("files", {}).get("html_url", "")

            # Additional debugging
            print(f"📊 Test result analysis:")
            print(f"   Success: {test_result.get('success')}")
            print(f"   Image URL: {image_url}")
            print(f"   HTML URL: {html_url}")
            print(f"   Files object: {test_result.get('files', {})}")

            return {
                "success": True,
                "message": "Image URL generation test successful",
                "test_data": {
                    "image_url": image_url,
                    "html_url": html_url,
                    "image_accessible": bool(image_url),
                    "html_accessible": bool(html_url),
                    "local_server_base_url": get_server_base_url() if 'get_server_base_url' in globals() else "Not available",
                    "remote_image_server_url": get_image_server_url() if 'get_image_server_url' in globals() else os.getenv("SERVER_URL", "Not available"),
                    "environment_variables": {
                        "SERVER_URL": server_url,
                        "BASE_URL": base_url
                    },
                    "url_analysis": {
                        "is_remote": image_url.startswith("https://") if image_url else False,
                        "is_local": image_url.startswith("http://localhost") if image_url else False,
                        "contains_server_url": server_url in image_url if server_url and image_url else False
                    }
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "error": test_result.get("error", "Graph generation failed"),
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        print(f"❌ Error testing image URLs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test image URLs: {str(e)}"
        )

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """Initialize and return the ChatGoogleGenerativeAI LLM instance."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

llm = initialize_llm_gemini()

# Database connection configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5433,
    'user': 'postgres',
    'password': '1234',
    'database': 'main_db'
}

def get_database_connection():
    """Create and return a database connection using the optimized connection pool."""
    return db_pool.get_connection()

async def get_db_id_from_user_id(user_id: str) -> int:
    """
    Fetch the current database ID for a user using direct database calls or HTTP endpoint.

    Args:
        user_id (str): User ID to get database ID for

    Returns:
        int: Database ID for the user

    Raises:
        Exception: If user has no current database or API call fails
    """
    try:
        print(f"🔍 Fetching database ID for user: {user_id}")

        # Try direct database manager call first
        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call (no HTTP overhead)")
            try:
                user_data = db_manager.get_user_current_db_details(user_id)
                if user_data and user_data.get('db_id'):
                    db_id = user_data['db_id']
                    print(f"✅ Found database ID {db_id} for user {user_id} via direct call")
                    return db_id
                else:
                    raise Exception(f"No current database found for user {user_id}")
            except Exception as e:
                print(f"❌ Direct database call failed: {e}")
                print("   Falling back to HTTP call...")

        # Fallback to HTTP call if direct call is not available or failed
        print("🌐 Using HTTP API call")
        import aiohttp

        # Load base URL from environment variable
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        print(f"🌐 Using base URL: {base_url}")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/mssql-config/user-current-db/{user_id}",
                timeout=aiohttp.ClientTimeout(total=30),
                ssl=False  # Disable SSL verification for self-signed certificates
            ) as response:

                if response.status == 200:
                    data = await response.json()
                    if data.get("status") == "success" and data.get("data"):
                        db_id = data["data"].get("db_id")
                        if db_id:
                            print(f"✅ Found database ID {db_id} for user {user_id} via HTTP")
                            return db_id
                        else:
                            raise Exception(f"No database ID found in response for user {user_id}")
                    else:
                        error_msg = data.get("message", "Unknown error")
                        raise Exception(f"API returned error: {error_msg}")
                elif response.status == 404:
                    raise Exception(f"No current database found for user {user_id}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")

    except Exception as e:
        print(f"❌ Error fetching database ID for user {user_id}: {e}")
        raise Exception(f"Failed to fetch database ID for user {user_id}: {str(e)}")

def get_db_id_from_user_id_sync(user_id: str) -> int:
    """
    Synchronous version of get_db_id_from_user_id.

    Args:
        user_id (str): User ID to get database ID for

    Returns:
        int: Database ID for the user

    Raises:
        Exception: If user has no current database or API call fails
    """
    return asyncio.run(get_db_id_from_user_id(user_id))

def get_report_structure_from_db(db_id: int = 1) -> str:
    """
    Fetch report_structure from the database for a given db_id with connection pooling.

    Args:
        db_id (int): Database ID (default: 1)

    Returns:
        str: Report structure content from database

    Raises:
        Exception: If database connection fails or record not found
    """
    conn = None
    try:
        print(f"\n🔍 DEBUG: Fetching report_structure from database ID {db_id}")

        # Try direct database manager call first
        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call (no SQL overhead)")
            try:
                db_config = db_manager.get_mssql_config(db_id)
                if db_config and db_config.get('report_structure'):
                    content = db_config['report_structure']
                    print(f"✅ Content retrieved successfully via direct call")
                    print(f"   Content length: {len(content)} characters")
                    print(f"   Content type: {type(content)}")

                    # Show first 500 characters as preview
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"   Content preview:")
                    print(f"   {'-'*50}")
                    print(f"   {preview}")
                    print(f"   {'-'*50}")

                    return content
                else:
                    print(f"❌ No report_structure found for database ID {db_id} via direct call")
                    raise Exception(f"Report structure not found for database ID {db_id}")
            except Exception as e:
                print(f"❌ Direct database call failed: {e}")
                print("   Falling back to SQL query...")

        # Fallback to direct SQL query with connection pooling
        print("🗄️ Using direct SQL query with connection pooling")
        conn = get_database_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT report_structure FROM mssql_config WHERE db_id = %s",
                (db_id,)
            )
            result = cursor.fetchone()

        print(f"✅ Database query completed")
        print(f"   Result found: {result is not None}")

        if result and result['report_structure']:
            content = result['report_structure']
            print(f"   Content retrieved successfully")
            print(f"   Content length: {len(content)} characters")
            print(f"   Content type: {type(content)}")

            # Show first 500 characters as preview
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"   Content preview:")
            print(f"   {'-'*50}")
            print(f"   {preview}")
            print(f"   {'-'*50}")

            return content
        else:
            print(f"❌ No report_structure found for database ID {db_id}")
            raise Exception(f"Report structure not found for database ID {db_id}")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        raise Exception(f"Database error: {str(e)}")
    except Exception as e:
        print(f"❌ Error fetching report structure: {e}")
        raise Exception(f"Failed to fetch report structure: {str(e)}")
    finally:
        if conn:
            db_pool.return_connection(conn)

def get_report_structure_from_user(user_id: str) -> str:
    """
    Fetch report_structure from the database for a given user_id.

    Args:
        user_id (str): User ID

    Returns:
        str: Report structure content from database

    Raises:
        Exception: If user has no current database, database connection fails, or record not found
    """
    try:
        # Get the database ID for the user
        db_id = cached_get_db_id_from_user_id_sync(user_id)

        # Fetch the report structure using the database ID
        return cached_get_report_structure_from_db(db_id)

    except Exception as e:
        print(f"❌ Error fetching report structure for user {user_id}: {e}")
        raise Exception(f"Failed to fetch report structure for user {user_id}: {str(e)}")

async def get_report_structure_from_user_async(user_id: str) -> str:
    """
    Async version of get_report_structure_from_user.

    Args:
        user_id (str): User ID

    Returns:
        str: Report structure content from database

    Raises:
        Exception: If user has no current database, database connection fails, or record not found
    """
    try:
        # Get the database ID for the user
        db_id = await get_db_id_from_user_id(user_id)

        # Fetch the report structure using the database ID
        return get_report_structure_from_db(db_id)

    except Exception as e:
        print(f"❌ Error fetching report structure for user {user_id}: {e}")
        raise Exception(f"Failed to fetch report structure for user {user_id}: {str(e)}")

def parse_report_structure_with_llm(report_structure_content: str, user_query: str = None) -> Dict[str, Any]:
    """
    Parse the report structure content using LLM to extract sections and queries.
    If user_query is provided, extract ALL queries but customize them based on the user's request.

    Args:
        report_structure_content (str): Report structure content from database
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")

    Returns:
        Dict[str, Any]: Structured data containing sections and their queries
        Format:
        {
            "sections": [
                {
                    "section_number": 1,
                    "section_name": "Employee Daily Attendance Summary for any Month",
                    "queries": [
                        {
                            "query_number": 1,
                            "query": "Get employee-wise punch count for March. 2024."
                        },
                        {
                            "query_number": 2,
                            "query": "Who are the users absent today on August 17, 2024..."
                        }
                    ]
                }
            ]
        }
    """

    # DEBUG: Print what is being sent to LLM
    print("\n" + "="*80)
    print("🔍 DEBUG: Content being sent to LLM from database:")
    print("="*80)
    if user_query:
        print(f"🎯 USER QUERY: {user_query}")
        print(f"📋 MODE: Extract ALL queries with customization based on user request")
    else:
        print(f"📋 MODE: Extract all queries (backward compatibility)")
    print(f"Content length: {len(report_structure_content)} characters")
    print(f"Content type: {type(report_structure_content)}")
    print("\n📄 Full content:")
    print("-"*80)
    print(report_structure_content)
    print("-"*80)
    print("="*80)

    # Create a context-aware prompt for the LLM
    if user_query:
        # User has provided a specific query - extract ALL queries but customize them based on user request
        prompt = f"""
        USER REQUEST: {user_query}

        Please analyze the following report structure content and extract ALL sections and queries, but customize the queries based on the user's request.

        Report Structure Content:
        {report_structure_content}

        Based on the user's request "{user_query}", please:
        1. Extract ALL sections and queries from the content
        2. If the user mentions specific names, dates, or values, incorporate them into the queries where appropriate
        3. If user mentions specific details, update the queries to include those details
        4. Keep all original queries but modify them to match the user's specific request

        Please return a JSON object with the following structure:
        {{
            "sections": [
                {{
                    "section_number": <number>,
                    "section_name": "<section name>",
                    "queries": [
                        {{
                            "query_number": <number>,
                            "query": "<query text - customized based on user request>"
                        }}
                    ]
                }}
            ]
        }}

        Rules:
        1. Extract section numbers from "### Section X:" or "### Section X:"
        2. Extract section names (the text after the colon)
        3. Extract ALL queries marked with "**Query:**" or "** Query:**"
        4. If user mentions specific details (names, dates, etc.), incorporate them into the queries
        5. DO NOT filter out any queries - include ALL queries from ALL sections
        6. If user asks for "financial report", customize queries to focus on financial data but keep all queries
        7. If user asks for "attendance report", customize queries to focus on attendance data but keep all queries
        8. If user asks for "general report", keep all queries as they are
        9. Number queries sequentially within each section
        10. Clean up any extra whitespace or formatting
        11. Return only valid JSON, no additional text

        IMPORTANT: Extract ALL queries from ALL sections, do not filter based on relevance.

        Please return the JSON structure:
        """
    else:
        # No user query - extract all content (backward compatibility)
        prompt = f"""
        Please parse the following report structure content and extract sections and queries in a structured format.

        Content:
        {report_structure_content}

        Please analyze this content and return a JSON object with the following structure:
        {{
            "sections": [
                {{
                    "section_number": <number>,
                    "section_name": "<section name>",
                    "queries": [
                        {{
                            "query_number": <number>,
                            "query": "<query text>"
                        }}
                    ]
                }}
            ]
        }}

        Rules:
        1. Extract section numbers from "### Section X:" or "### Section X:"
        2. Extract section names (the text after the colon)
        3. Extract all queries marked with "**Query:**" or "** Query:**"
        4. Number queries sequentially within each section
        5. Clean up any extra whitespace or formatting
        6. Return only valid JSON, no additional text

        Please return the JSON structure:
        """

    # DEBUG: Print the prompt being sent to LLM
    print("\n" + "="*80)
    print("🤖 DEBUG: Prompt being sent to LLM:")
    print("="*80)
    print(prompt)
    print("="*80)

    try:
        # Get response from LLM
        response = llm.invoke(prompt)
        response_text = response.content

        # DEBUG: Print LLM response
        print("\n" + "="*80)
        print("🤖 DEBUG: LLM Response:")
        print("="*80)
        print(response_text)
        print("="*80)

        # Extract JSON from response (in case LLM adds extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)

            # DEBUG: Print parsed data
            print("\n" + "="*80)
            print("✅ DEBUG: Parsed JSON Data:")
            print("="*80)
            print(json.dumps(parsed_data, indent=2))
            print("="*80)

            return parsed_data
        else:
            # If no JSON found, try to parse the entire response
            parsed_data = json.loads(response_text)

            # DEBUG: Print parsed data
            print("\n" + "="*80)
            print("✅ DEBUG: Parsed JSON Data (from full response):")
            print("="*80)
            print(json.dumps(parsed_data, indent=2))
            print("="*80)

            return parsed_data

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing LLM response as JSON: {e}")
        print(f"Raw response: {response_text}")
        raise Exception("Failed to parse LLM response as valid JSON")
    except Exception as e:
        print(f"❌ Error getting response from LLM: {e}")
        raise Exception(f"LLM processing failed: {str(e)}")

# Keep the old function for backward compatibility
def parse_testing_file_with_llm(file_path: str = "Report_generator/utilites/testing.txt") -> Dict[str, Any]:
    """
    Parse the testing.txt file using LLM to extract sections and queries.
    This function is kept for backward compatibility.

    Args:
        file_path (str): Path to the testing.txt file

    Returns:
        Dict[str, Any]: Structured data containing sections and their queries
    """

    # Read the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

    # Use the new parsing function
    return parse_report_structure_with_llm(content)

def get_sections_and_queries_from_db(db_id: int = 1) -> str:
    """
    Get sections and queries from the database and return formatted string.

    Args:
        db_id (int): Database ID (default: 1)

    Returns:
        str: Formatted string with sections and queries
    """
    try:
        # Fetch report structure from database
        report_structure_content = cached_get_report_structure_from_db(db_id)

        # Parse the content
        parsed_data = parse_report_structure_with_llm(report_structure_content)

        # Format the output as requested
        output_lines = []

        for section in parsed_data.get("sections", []):
            section_num = section.get("section_number", "Unknown")
            section_name = section.get("section_name", "Unknown Section")

            output_lines.append(f"Section {section_num}: {section_name}")

            for query in section.get("queries", []):
                query_num = query.get("query_number", "Unknown")
                query_text = query.get("query", "No query text")
                output_lines.append(f"  Query {query_num}: {query_text}")

            output_lines.append("")  # Add empty line between sections

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error processing database content: {str(e)}"

def get_sections_and_queries_dict_from_db(db_id: int = 1, user_query: str = None) -> Dict[str, Any]:
    """
    Get sections and queries from the database and return as dictionary.
    If user_query is provided, return ALL queries but customize them based on the user's request.

    Args:
        db_id (int): Database ID (default: 1)
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")

    Returns:
        Dict[str, Any]: Structured data with sections and queries
    """
    try:
        # Fetch report structure from database
        report_structure_content = cached_get_report_structure_from_db(db_id)

        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)

    except Exception as e:
        print(f"Error getting sections and queries from database: {e}")
        return {"sections": [], "error": str(e)}

def get_sections_and_queries_from_user(user_id: str) -> str:
    """
    Get sections and queries from the database for a user and return formatted string.

    Args:
        user_id (str): User ID

    Returns:
        str: Formatted string with sections and queries
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = get_report_structure_from_user(user_id)

        # Parse the content
        parsed_data = parse_report_structure_with_llm(report_structure_content)

        # Format the output as requested
        output_lines = []

        for section in parsed_data.get("sections", []):
            section_num = section.get("section_number", "Unknown")
            section_name = section.get("section_name", "Unknown Section")

            output_lines.append(f"Section {section_num}: {section_name}")

            for query in section.get("queries", []):
                query_num = query.get("query_number", "Unknown")
                query_text = query.get("query", "No query text")
                output_lines.append(f"  Query {query_num}: {query_text}")

            output_lines.append("")  # Add empty line between sections

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error processing database content for user {user_id}: {str(e)}"

def get_sections_and_queries_dict_from_user(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    Get sections and queries from the database for a user and return as dictionary.
    If user_query is provided, return ALL queries but customize them based on the user's request.

    Args:
        user_id (str): User ID
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")

    Returns:
        Dict[str, Any]: Structured data with sections and queries
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = get_report_structure_from_user(user_id)

        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)

    except Exception as e:
        print(f"Error getting sections and queries from database for user {user_id}: {e}")
        return {"sections": [], "error": str(e)}

async def get_sections_and_queries_dict_from_user_async(user_id: str, user_query: str = None) -> Dict[str, Any]:
    """
    Async version of get_sections_and_queries_dict_from_user.

    Args:
        user_id (str): User ID
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary")

    Returns:
        Dict[str, Any]: Structured data with sections and queries (ALL queries, customized based on user request)
    """
    try:
        # Fetch report structure from database for the user
        report_structure_content = await get_report_structure_from_user_async(user_id)

        # Parse the content with user query if provided
        return parse_report_structure_with_llm(report_structure_content, user_query)

    except Exception as e:
        print(f"Error getting sections and queries from database for user {user_id}: {e}")
        return {"sections": [], "error": str(e)}

async def process_all_queries_with_graph_generation(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Process all queries from the database by calling the generate-graph function directly.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Results for all processed queries
    """

    try:
        print(f"🚀 Starting batch processing of all queries from database ID {db_id}")

        # Step 1: Parse the database content
        print("📖 Parsing database content...")
        sections_data = get_sections_and_queries_dict_from_db(db_id)

        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse database content or no sections found")

        # Step 2: Extract all queries
        all_queries = []
        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")

            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")

                all_queries.append({
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                })

        print(f"📊 Found {len(all_queries)} queries to process")

        # Step 3: Process each query using direct function calls
        results = []
        successful_count = 0
        failed_count = 0

        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using direct graph generation functions (no HTTP overhead)")

            async def process_single_query_direct(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query using direct function call."""
                query_start_time = time.time()

                try:
                    # Call the direct graph generation function
                    graph_result = await generate_beautiful_graph(
                        query=query_info["query"],
                        user_id=user_id,
                        export_format=export_format,
                        theme=theme
                    )

                    query_processing_time = time.time() - query_start_time

                    if graph_result.get("success"):
                        # Serialize the graph_result to handle date objects
                        serialized_graph_result = serialize_for_json(graph_result)

                        # Get the data for pagination and analysis
                        data = serialized_graph_result.get("data", {}).get("sample", [])

                        # Create simple table with all data
                        table = create_simple_table(data)

                        # Create graph analysis with LLM data analysis
                        graph_analysis = create_graph_analysis(graph_result, data, query_info["query"])

                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": True,
                            "table": table,
                            "graph_and_analysis": graph_analysis
                        }
                    else:
                        return {
                            "section_number": query_info["section_number"],
                            "section_name": query_info["section_name"],
                            "query_number": query_info["query_number"],
                            "query": query_info["query"],
                            "success": False,
                            "error": str(graph_result.get("error", "Unknown error"))
                        }

                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }

            # Process queries with limited concurrency to avoid overwhelming the system
            semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests

            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_direct(query_info)

            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        else:
            # Fallback to HTTP API calls if direct functions are not available
            print("⚠️ Using HTTP API calls (direct functions not available)")
            import aiohttp

            # Load base URL from environment variable
            base_url = os.getenv("BASE_URL", "http://localhost:8000")
            print(f"🌐 Using base URL: {base_url}")

            async def process_single_query_http(query_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single query by calling the generate-graph endpoint."""
                query_start_time = time.time()

                try:
                    # Prepare request payload
                    payload = {
                        "query": query_info["query"],
                        "user_id": user_id,
                        "export_format": export_format,
                        "theme": theme,
                        "analysis_subject": analysis_subject
                    }

                    # Make async HTTP request
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{base_url}/graph/generate-graph",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=300),  # 5 minutes timeout
                            ssl=False  # Disable SSL verification for self-signed certificates
                        ) as response:

                            if response.status == 200:
                                result_data = await response.json()
                                query_processing_time = time.time() - query_start_time

                                # Serialize the result_data to handle date objects
                                serialized_result_data = serialize_for_json(result_data)

                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": True,
                                    "files": serialized_result_data.get("files", {}),
                                    "data": serialized_result_data.get("data", {})
                                }
                            else:
                                error_text = await response.text()
                                query_processing_time = time.time() - query_start_time

                                return {
                                    "section_number": query_info["section_number"],
                                    "section_name": query_info["section_name"],
                                    "query_number": query_info["query_number"],
                                    "query": query_info["query"],
                                    "success": False,
                                    "error": f"HTTP {response.status}: {error_text}"
                                }

                except Exception as e:
                    query_processing_time = time.time() - query_start_time
                    return {
                        "section_number": query_info["section_number"],
                        "section_name": query_info["section_name"],
                        "query_number": query_info["query_number"],
                        "query": query_info["query"],
                        "success": False,
                        "error": str(e)
                    }

            # Process queries concurrently (limit to 5 concurrent requests to avoid overwhelming the server)
            semaphore = asyncio.Semaphore(5)

            async def process_with_semaphore(query_info):
                async with semaphore:
                    return await process_single_query_http(query_info)

            # Execute all queries
            tasks = [process_with_semaphore(query_info) for query_info in all_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_count = sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
        failed_count = len(results) - successful_count

        # Generate summary
        total_processing_time = sum(r.get("processing_time", 0) for r in results if isinstance(r, dict))

        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results if isinstance(r, dict))),
            "processing_method": "direct_function" if DIRECT_GRAPH_AVAILABLE else "http_api",
            "database_id": db_id,
            "errors_summary": {}
        }

        # Group errors by type
        for result in results:
            if isinstance(result, dict) and result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1

        print(f"🎉 Batch processing completed!")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: {'Direct function calls' if DIRECT_GRAPH_AVAILABLE else 'HTTP API calls'}")

        return {
            "success": True,
            "database_id": db_id,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

# Keep the old function for backward compatibility
async def process_all_queries_with_graph_generation_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Process all queries from the testing file by calling the generate-graph function directly.
    This function is kept for backward compatibility.

    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Results for all processed queries
    """

    try:
        print(f"🚀 Starting OPTIMIZED batch processing of all queries from {file_path}")

        # Step 1: Parse the testing file
        print("📖 Parsing testing file...")
        sections_data = get_sections_and_queries_dict_from_db(file_path)

        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse testing file or no sections found")

        # Step 2: Extract all queries
        all_queries = []
        query_mapping = {}  # Map query text to section info

        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")

            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")

                all_queries.append(query_text)
                query_mapping[query_text] = {
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                }

        print(f"📊 Found {len(all_queries)} queries to process")

        # Step 3: Use the optimized batch processing function
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using optimized generate_multiple_graphs function")
            start_time = time.time()

            # Call the optimized batch function
            graph_results = await generate_multiple_graphs(
                queries=all_queries,
                user_id=user_id,
                theme=theme,
                export_format=export_format
            )

            total_processing_time = time.time() - start_time

            # Step 4: Process results and map back to section information
            results = []
            successful_count = 0
            failed_count = 0

            for i, graph_result in enumerate(graph_results):
                query_text = all_queries[i]
                section_info = query_mapping[query_text]

                if graph_result.get("success"):
                    successful_count += 1

                    # Serialize the graph_result to handle date objects
                    serialized_graph_result = serialize_for_json(graph_result)

                    # Get the data for pagination and analysis
                    data = serialized_graph_result.get("data", {}).get("sample", [])

                    # Create simple table with all data
                    table = create_simple_table(data)

                    # Create graph analysis with LLM data analysis
                    graph_analysis = create_graph_analysis(graph_result, data, section_info["query"])

                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": True,
                        "table": table,
                        "graph_and_analysis": graph_analysis
                    })
                else:
                    failed_count += 1
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": False,
                        "error": str(graph_result.get("error", "Unknown error"))
                    })

        else:
            # Fallback to the regular function if direct functions are not available
            print("⚠️ Direct functions not available, falling back to regular processing")
            return await process_all_queries_with_graph_generation_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )

        # Generate summary
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results)),
            "processing_method": "optimized_batch_function",
            "database_id": db_id,
            "errors_summary": {}
        }

        # Group errors by type
        for result in results:
            if result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1

        print(f"🎉 OPTIMIZED batch processing completed!")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: Optimized batch function")

        return {
            "success": True,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in optimized batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

async def process_all_queries_with_graph_generation_optimized(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    user_query: str = None,
    task_id: str = None  # Add task_id parameter
) -> Dict[str, Any]:
    """
    OPTIMIZED VERSION: Process all queries using the generate_multiple_graphs function.
    This is the most efficient method as it's specifically designed for batch processing.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        user_query (str, optional): User's specific request for filtering queries
        task_id (str, optional): Task ID for tracking current query processing

    Returns:
        Dict[str, Any]: Results for all processed queries
    """

    try:
        if user_query:
            print(f"🚀 Starting OPTIMIZED batch processing of ALL queries with customization from database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"🚀 Starting OPTIMIZED batch processing of all queries from database ID {db_id}")

        # Step 1: Parse the database content
        if user_query:
            print(f"📖 Parsing database content with user query: {user_query}")
        else:
            print("📖 Parsing database content...")
        sections_data = get_sections_and_queries_dict_from_db(db_id, user_query)

        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse database content or no sections found")

        # Step 2: Extract all queries
        all_queries = []
        query_mapping = {}  # Map query text to section info

        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")

            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")

                all_queries.append(query_text)
                query_mapping[query_text] = {
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                }

        print(f"📊 Found {len(all_queries)} queries to process")

        # Step 3: Use the optimized batch processing function
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using optimized generate_multiple_graphs function")
            start_time = time.time()

            # Track processed queries for progress updates
            processed_count = 0
            total_queries = len(all_queries)
            
            # Update task with total queries if task_id is provided
            if task_id:
                report_task_manager.update_task(
                    task_id,
                    total_queries=total_queries,
                    processed_queries=0,
                    successful_queries=0,
                    failed_queries=0
                )

            # Process queries one by one to track current query
            results = []
            successful_count = 0
            failed_count = 0

            for i, query_text in enumerate(all_queries):
                current_query_num = i + 1
                section_info = query_mapping[query_text]
                
                # Update task with current query being processed
                if task_id:
                    report_task_manager.update_task(
                        task_id,
                        current_query=query_text,
                        progress=f"Processing query {current_query_num}/{total_queries}",
                        current_step=f"Processing: {query_text[:50]}{'...' if len(query_text) > 50 else ''}",
                        processed_queries=processed_count
                    )
                
                print(f"🔄 Processing query {current_query_num}/{total_queries}: {query_text[:50]}{'...' if len(query_text) > 50 else ''}")
                
                try:
                    # Process single query
                    graph_result = await generate_beautiful_graph(
                        query=query_text,
                        user_id=user_id,
                        theme=theme,
                        export_format=export_format
                    )
                    
                    if graph_result.get("success"):
                        successful_count += 1
                        processed_count += 1

                        # Serialize the graph_result to handle date objects
                        serialized_graph_result = serialize_for_json(graph_result)

                        # Get the data for pagination and analysis
                        data = serialized_graph_result.get("data", {}).get("sample", [])

                        # Create simple table with all data
                        table = create_simple_table(data)

                        # Create enhanced graph analysis with comprehensive column analysis
                        graph_analysis = create_graph_analysis(graph_result, data, section_info["query"])

                        # Extract key insights for summary
                        column_insights = extract_key_column_insights(graph_analysis)

                        results.append({
                            "section_number": section_info["section_number"],
                            "section_name": section_info["section_name"],
                            "query_number": section_info["query_number"],
                            "query": section_info["query"],
                            "success": True,
                            "table": table,
                            "graph_and_analysis": graph_analysis,
                            "column_insights": column_insights  # Add key column insights
                        })
                        
                        # Update task with progress
                        if task_id:
                            report_task_manager.update_task(
                                task_id,
                                processed_queries=processed_count,
                                successful_queries=successful_count,
                                failed_queries=failed_count
                            )
                    else:
                        failed_count += 1
                        processed_count += 1
                        results.append({
                            "section_number": section_info["section_number"],
                            "section_name": section_info["section_name"],
                            "query_number": section_info["query_number"],
                            "query": section_info["query"],
                            "success": False,
                            "error": str(graph_result.get("error", "Unknown error"))
                        })
                        
                        # Update task with progress and error
                        if task_id:
                            report_task_manager.update_task(
                                task_id,
                                processed_queries=processed_count,
                                successful_queries=successful_count,
                                failed_queries=failed_count
                            )
                        
                except Exception as e:
                    failed_count += 1
                    processed_count += 1
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": False,
                        "error": str(e)
                    })
                    
                    # Update task with progress and error
                    if task_id:
                        report_task_manager.update_task(
                            task_id,
                            processed_queries=processed_count,
                            successful_queries=successful_count,
                            failed_queries=failed_count
                        )
                    print(f"❌ Error processing query {current_query_num}: {e}")

            total_processing_time = time.time() - start_time

        else:
            # Fallback to the regular function if direct functions are not available
            print("⚠️ Direct functions not available, falling back to regular processing")
            return await process_all_queries_with_graph_generation(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )

        # Generate enhanced summary with column analysis insights
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results)),
            "processing_method": "optimized_batch_function",
            "database_id": db_id,
            "errors_summary": {},
            "column_analysis_summary": generate_column_analysis_summary(results)
        }

        # Group errors by type
        for result in results:
            if result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1

        print(f"🎉 OPTIMIZED batch processing completed!")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: Optimized batch function")

        # Clear current query when done
        if task_id:
            report_task_manager.update_task(
                task_id,
                current_query=None
            )

        return {
            "success": True,
            "database_id": db_id,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in optimized batch processing: {e}")
        # Clear current query on error
        if task_id:
            report_task_manager.update_task(
                task_id,
                current_query=None
            )
        return {
            "success": False,
            "error": str(e),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

def process_all_queries_with_graph_generation_sync(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Synchronous wrapper for process_all_queries_with_graph_generation.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    return asyncio.run(process_all_queries_with_graph_generation(
        db_id=db_id,
        user_id=user_id,
        export_format=export_format,
        theme=theme,
        analysis_subject=analysis_subject
    ))

def process_all_queries_with_graph_generation_optimized_sync(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    user_query: str = None,
    task_id: str = None  # Add task_id parameter
) -> Dict[str, Any]:
    """
    Synchronous wrapper for process_all_queries_with_graph_generation_optimized.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        user_query (str, optional): User's specific request for filtering queries
        task_id (str, optional): Task ID for tracking current query processing

    Returns:
        Dict[str, Any]: Results for all processed queries
    """
    return asyncio.run(process_all_queries_with_graph_generation_optimized(
        db_id=db_id,
        user_id=user_id,
        export_format=export_format,
        theme=theme,
        analysis_subject=analysis_subject,
        user_query=user_query,
        task_id=task_id  # Pass task_id
    ))

# Keep the old functions for backward compatibility
async def process_all_queries_with_graph_generation_optimized_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    OPTIMIZED VERSION: Process all queries using the generate_multiple_graphs function.
    This function is kept for backward compatibility.

    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Results for all processed queries
    """

    try:
        print(f"🚀 Starting OPTIMIZED batch processing of all queries from {file_path}")

        # Step 1: Parse the testing file
        print("📖 Parsing testing file...")
        sections_data = get_sections_and_queries_dict_from_db(file_path)

        if not sections_data or "sections" not in sections_data:
            raise Exception("Failed to parse testing file or no sections found")

        # Step 2: Extract all queries
        all_queries = []
        query_mapping = {}  # Map query text to section info

        for section in sections_data["sections"]:
            section_num = section.get("section_number", 0)
            section_name = section.get("section_name", "Unknown Section")

            for query in section.get("queries", []):
                query_num = query.get("query_number", 0)
                query_text = query.get("query", "")

                all_queries.append(query_text)
                query_mapping[query_text] = {
                    "section_number": section_num,
                    "section_name": section_name,
                    "query_number": query_num,
                    "query": query_text
                }

        print(f"📊 Found {len(all_queries)} queries to process")

        # Step 3: Use the optimized batch processing function
        if DIRECT_GRAPH_AVAILABLE:
            print("🎯 Using optimized generate_multiple_graphs function")
            start_time = time.time()

            # Call the optimized batch function
            graph_results = await generate_multiple_graphs(
                queries=all_queries,
                user_id=user_id,
                theme=theme,
                export_format=export_format
            )

            total_processing_time = time.time() - start_time

            # Step 4: Process results and map back to section information
            results = []
            successful_count = 0
            failed_count = 0

            for i, graph_result in enumerate(graph_results):
                query_text = all_queries[i]
                section_info = query_mapping[query_text]

                if graph_result.get("success"):
                    successful_count += 1

                    # Serialize the graph_result to handle date objects
                    serialized_graph_result = serialize_for_json(graph_result)

                    # Get the data for pagination and analysis
                    data = serialized_graph_result.get("data", {}).get("sample", [])

                    # Create simple table with all data
                    table = create_simple_table(data)

                    # Create graph analysis with LLM data analysis
                    graph_analysis = create_graph_analysis(graph_result, data, section_info["query"])

                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": True,
                        "table": table,
                        "graph_and_analysis": graph_analysis
                    })
                else:
                    failed_count += 1
                    results.append({
                        "section_number": section_info["section_number"],
                        "section_name": section_info["section_name"],
                        "query_number": section_info["query_number"],
                        "query": section_info["query"],
                        "success": False,
                        "error": str(graph_result.get("error", "Unknown error"))
                    })

        else:
            # Fallback to the regular function if direct functions are not available
            print("⚠️ Direct functions not available, falling back to regular processing")
            return await process_all_queries_with_graph_generation_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )

        # Generate summary
        summary = {
            "total_sections": len(sections_data["sections"]),
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "success_rate": (successful_count / len(all_queries)) * 100 if all_queries else 0,
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(all_queries) if all_queries else 0,
            "sections_processed": len(set(r.get("section_number", 0) for r in results)),
            "processing_method": "optimized_batch_function",
            "database_id": db_id,
            "errors_summary": {}
        }

        # Group errors by type
        for result in results:
            if result.get("error"):
                error_type = result["error"].split(":")[0] if ":" in result["error"] else "Unknown"
                summary["errors_summary"][error_type] = summary["errors_summary"].get(error_type, 0) + 1

        print(f"🎉 OPTIMIZED batch processing completed!")
        print(f"   Total queries: {len(all_queries)}")
        print(f"   Successful: {successful_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Total time: {total_processing_time:.2f}s")
        print(f"   Method: Optimized batch function")

        return {
            "success": True,
            "total_queries": len(all_queries),
            "successful_queries": successful_count,
            "failed_queries": failed_count,
            "results": results,
            "total_processing_time": total_processing_time,
            "summary": summary
        }

    except Exception as e:
        print(f"❌ Error in optimized batch processing: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "total_processing_time": 0,
            "summary": {}
        }

def serialize_for_json(obj, exclude_html=True):
    """
    Enhanced JSON serializer to handle date objects and other non-serializable types.
    Excludes HTML content from graph results to reduce response size.

    Args:
        obj: Object to serialize
        exclude_html: Whether to exclude HTML content from graph results
    """
    if obj is None:
        return None
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif hasattr(obj, 'isoformat'):  # For any object with isoformat method
        return obj.isoformat()
    elif hasattr(obj, 'strftime'):  # For any object with strftime method
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif hasattr(obj, 'tolist'):  # For numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        # Special handling for graph results to exclude HTML content
        if exclude_html and 'graph' in obj and isinstance(obj['graph'], dict):
            # Create a clean version of graph data without HTML
            clean_graph = {}
            for k, v in obj['graph'].items():
                if k != 'html':  # Exclude HTML content
                    clean_graph[k] = serialize_for_json(v, exclude_html)
            # Add a flag indicating HTML was removed
            clean_graph['html_removed'] = True

            # Create new dict with clean graph
            result = {}
            for k, v in obj.items():
                if k == 'graph':
                    result[k] = clean_graph
                else:
                    result[k] = serialize_for_json(v, exclude_html)
            return result
        else:
            return {str(k): serialize_for_json(v, exclude_html) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item, exclude_html) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {str(k): serialize_for_json(v, exclude_html) for k, v in obj.__dict__.items()}
    else:
        return str(obj)

def create_simple_table(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a simple table structure from data without pagination.

    Args:
        data: List of dictionaries containing the data

    Returns:
        Dict containing simple table structure with all data
    """
    if not data:
        return {
            "total_rows": 0,
            "columns": [],
            "data": []
        }

    total_rows = len(data)
    columns = list(data[0].keys()) if data else []

    return {
        "total_rows": total_rows,
        "columns": columns,
        "data": serialize_for_json(data)
    }

def create_graph_analysis(graph_result: Dict[str, Any], data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """
    Create comprehensive analysis and metadata for the graph including column analysis.

    Args:
        graph_result: Result from graph generation
        data: The data used for the graph
        query: The original query

    Returns:
        Dict containing comprehensive graph analysis with column insights
    """
    if not graph_result.get("success"):
        return {
            "error": graph_result.get("error", "Graph generation failed"),
            "status": "failed"
        }

    # Extract graph information
    graph_type = graph_result.get("graph", {}).get("type", "unknown")
    image_url = graph_result.get("files", {}).get("image_url", "")

    # Basic data analysis
    total_records = len(data)
    columns_count = len(data[0].keys()) if data else 0
    sql_query = graph_result.get("data", {}).get("sql", "")
    generation_time = graph_result.get("metadata", {}).get("generation_time", datetime.now().isoformat())

    # Create data summary
    data_summary = f"Generated {graph_type.title()} chart with {total_records} records across {columns_count} columns"

    # Extract LLM data analysis if available
    llm_analysis = graph_result.get("data_analysis", {})

    # Extract comprehensive metadata from graph generation
    metadata = graph_result.get("metadata", {})

    # Enhanced column analysis structure
    column_analysis = {
        "business_context": metadata.get("business_context", {}),
        "column_semantics": metadata.get("semantics", {}),
        "column_mapping": metadata.get("column_mapping", {}),
        "data_transformations": metadata.get("data_transformations", {}),
        "visualization_config": metadata.get("visualization_config", {}),
        "performance_considerations": metadata.get("performance_considerations", {}),
        "reasoning": metadata.get("reasoning", {}),
        "quality_assessment": metadata.get("quality_assessment", {})
    }

    # Extract column recommendations for visualization
    column_recommendations = extract_column_recommendations(column_analysis, data)

    # Create enhanced analysis structure
    enhanced_analysis = {
        "graph_type": graph_type,
        "theme": graph_result.get("graph", {}).get("theme", "modern"),
        "image_url": image_url,
        "column_mapping": metadata.get("column_mapping", {}),  # Column mapping at top level
        "analysis": {
            "total_records": total_records,
            "columns_count": columns_count,
            "sql_query": sql_query,
            "generation_time": generation_time,
            "data_summary": data_summary,
            "query": query
        },
        "llm_analysis": llm_analysis,  # Include LLM analysis from graph_Generator
        "column_analysis": column_analysis,  # Comprehensive column analysis
        "column_recommendations": column_recommendations,  # Specific recommendations
        "visualization_insights": {
            "recommended_chart_type": graph_type,
            "business_domain": column_analysis.get("business_context", {}).get("domain", "unknown"),
            "analysis_intent": column_analysis.get("business_context", {}).get("intent", "unknown"),
            "decision_level": column_analysis.get("business_context", {}).get("decision_level", "unknown"),
            "confidence_score": column_analysis.get("quality_assessment", {}).get("confidence_score", 0),
            "data_quality": column_analysis.get("quality_assessment", {}).get("data_quality", "unknown"),
            "visualization_fit": column_analysis.get("quality_assessment", {}).get("visualization_fit", "unknown")
        }
    }

    return enhanced_analysis


def extract_column_recommendations(column_analysis: Dict[str, Any], data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract specific column recommendations for visualization and analysis.

    Args:
        column_analysis: Comprehensive column analysis from graph generation
        data: The actual data used for the graph

    Returns:
        Dict containing specific recommendations for each column
    """
    if not data:
        return {}

    columns = list(data[0].keys()) if data else []
    semantics = column_analysis.get("column_semantics", {})
    column_mapping = column_analysis.get("column_mapping", {})

    recommendations = {
        "primary_columns": {},
        "secondary_columns": {},
        "excluded_columns": {},
        "visualization_suggestions": [],
        "data_quality_insights": [],
        "business_insights": []
    }

    # Analyze each column
    for column in columns:
        column_info = semantics.get(column, {})

        # Determine column role in visualization
        column_role = "excluded"
        for role, mapped_column in column_mapping.items():
            if mapped_column == column:
                column_role = role
                break

        # Create column recommendation
        column_recommendation = {
            "column_name": column,
            "role": column_role,
            "business_meaning": column_info.get("meaning", "Unknown"),
            "category": column_info.get("category", "unknown"),
            "quality_score": column_info.get("quality_score", 0),
            "outlier_risk": column_info.get("outlier_risk", "unknown"),
            "priority": column_info.get("priority", "low"),
            "kpi_type": column_info.get("kpi_type", "unknown"),
            "data_characteristics": column_info.get("data_characteristics", "unknown"),
            "recommended_usage": get_column_usage_recommendation(column_role, column_info),
            "visualization_suitability": get_visualization_suitability(column_info)
        }

        # Categorize columns
        if column_role in ["x", "y", "labels", "values"]:
            recommendations["primary_columns"][column] = column_recommendation
        elif column_role in ["color", "size", "text", "group"]:
            recommendations["secondary_columns"][column] = column_recommendation
        else:
            recommendations["excluded_columns"][column] = column_recommendation

    # Add visualization suggestions
    reasoning = column_analysis.get("reasoning", {})
    if reasoning:
        recommendations["visualization_suggestions"] = [
            {
                "suggestion": "Chart Type Selection",
                "explanation": reasoning.get("why_chart", "No explanation provided"),
                "business_value": reasoning.get("business_value", "No business value specified")
            },
            {
                "suggestion": "Column Selection",
                "explanation": reasoning.get("why_columns", "No explanation provided"),
                "user_benefits": reasoning.get("user_benefits", "No user benefits specified")
            }
        ]

    # Add data quality insights
    for column, info in semantics.items():
        if info.get("quality_score", 0) < 5:
            recommendations["data_quality_insights"].append({
                "column": column,
                "issue": "Low data quality score",
                "score": info.get("quality_score", 0),
                "recommendation": "Consider data cleaning or alternative columns"
            })

    # Add business insights
    business_context = column_analysis.get("business_context", {})
    if business_context:
        recommendations["business_insights"] = [
            {
                "domain": business_context.get("domain", "unknown"),
                "intent": business_context.get("intent", "unknown"),
                "decision_level": business_context.get("decision_level", "unknown"),
                "stakeholders": business_context.get("stakeholders", "unknown")
            }
        ]

    return recommendations


def get_column_usage_recommendation(role: str, column_info: Dict[str, Any]) -> str:
    """
    Get specific usage recommendation for a column based on its role and characteristics.

    Args:
        role: Column role in visualization (x, y, color, etc.)
        column_info: Column semantic information

    Returns:
        String recommendation for column usage
    """
    category = column_info.get("category", "unknown")
    quality_score = column_info.get("quality_score", 0)

    if role == "x":
        if category == "temporal":
            return "Excellent for time-series analysis and trend visualization"
        elif category == "categorical":
            return "Good for grouping and comparison across categories"
        else:
            return "Suitable for x-axis positioning"

    elif role == "y":
        if category == "financial":
            return "Perfect for financial metrics and KPI visualization"
        elif category == "metric":
            return "Ideal for quantitative analysis and performance measurement"
        else:
            return "Appropriate for y-axis values"

    elif role == "color":
        if category == "categorical":
            return "Excellent for categorical differentiation and grouping"
        else:
            return "Good for visual distinction and pattern recognition"

    elif role == "size":
        if category == "metric":
            return "Perfect for bubble charts and size-based emphasis"
        else:
            return "Suitable for proportional representation"

    else:
        return "Consider for additional context and detail"


def get_visualization_suitability(column_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess visualization suitability for a column.

    Args:
        column_info: Column semantic information

    Returns:
        Dict with suitability assessment
    """
    category = column_info.get("category", "unknown")
    quality_score = column_info.get("quality_score", 0)
    data_characteristics = column_info.get("data_characteristics", "unknown")

    suitability = {
        "overall_score": quality_score,
        "recommended_charts": [],
        "avoid_charts": [],
        "data_limitations": []
    }

    # Chart recommendations based on category
    if category == "temporal":
        suitability["recommended_charts"] = ["line", "area", "candlestick", "step"]
        suitability["avoid_charts"] = ["pie", "donut"]
    elif category == "categorical":
        suitability["recommended_charts"] = ["bar", "column", "pie", "donut", "treemap"]
        suitability["avoid_charts"] = ["line", "scatter"]
    elif category == "financial":
        suitability["recommended_charts"] = ["bar", "column", "line", "waterfall", "candlestick"]
        suitability["avoid_charts"] = ["pie", "donut"]
    elif category == "metric":
        suitability["recommended_charts"] = ["bar", "column", "line", "scatter", "bubble"]
        suitability["avoid_charts"] = ["pie", "donut"]

    # Data limitations
    if quality_score < 5:
        suitability["data_limitations"].append("Low data quality - consider data cleaning")
    if data_characteristics == "discrete":
        suitability["data_limitations"].append("Discrete data - may limit continuous visualizations")

    return suitability


def extract_key_column_insights(graph_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key column insights for summary display.

    Args:
        graph_analysis: Enhanced graph analysis with column information

    Returns:
        Dict containing key insights for quick reference
    """
    column_analysis = graph_analysis.get("column_analysis", {})
    column_recommendations = graph_analysis.get("column_recommendations", {})
    visualization_insights = graph_analysis.get("visualization_insights", {})

    # Extract key metrics
    primary_columns = list(column_recommendations.get("primary_columns", {}).keys())
    secondary_columns = list(column_recommendations.get("secondary_columns", {}).keys())
    excluded_columns = list(column_recommendations.get("excluded_columns", {}).keys())

    # Get data quality insights
    data_quality_issues = column_recommendations.get("data_quality_insights", [])

    # Get business context
    business_context = column_analysis.get("business_context", {})

    # Create summary insights
    key_insights = {
        "chart_type": graph_analysis.get("graph_type", "unknown"),
        "business_domain": business_context.get("domain", "unknown"),
        "analysis_intent": business_context.get("intent", "unknown"),
        "confidence_score": visualization_insights.get("confidence_score", 0),
        "data_quality": visualization_insights.get("data_quality", "unknown"),
        "column_summary": {
            "total_columns": len(primary_columns) + len(secondary_columns) + len(excluded_columns),
            "primary_columns": primary_columns,
            "secondary_columns": secondary_columns,
            "excluded_columns": excluded_columns
        },
        "data_quality_issues": len(data_quality_issues),
        "key_recommendations": extract_top_recommendations(column_recommendations),
        "business_insights": extract_business_insights(business_context, column_analysis)
    }

    return key_insights


def extract_top_recommendations(column_recommendations: Dict[str, Any]) -> List[str]:
    """
    Extract top recommendations from column analysis.

    Args:
        column_recommendations: Column recommendations dictionary

    Returns:
        List of top recommendations
    """
    recommendations = []

    # Add visualization suggestions
    viz_suggestions = column_recommendations.get("visualization_suggestions", [])
    for suggestion in viz_suggestions[:2]:  # Top 2 suggestions
        recommendations.append(f"{suggestion['suggestion']}: {suggestion['explanation'][:100]}...")

    # Add data quality recommendations
    quality_insights = column_recommendations.get("data_quality_insights", [])
    for insight in quality_insights[:2]:  # Top 2 quality issues
        recommendations.append(f"Data Quality: {insight['column']} - {insight['recommendation']}")

    return recommendations


def extract_business_insights(business_context: Dict[str, Any], column_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract business insights from analysis.

    Args:
        business_context: Business context information
        column_analysis: Column analysis data

    Returns:
        Dict containing business insights
    """
    reasoning = column_analysis.get("reasoning", {})

    return {
        "domain": business_context.get("domain", "unknown"),
        "intent": business_context.get("intent", "unknown"),
        "decision_level": business_context.get("decision_level", "unknown"),
        "stakeholders": business_context.get("stakeholders", "unknown"),
        "business_value": reasoning.get("business_value", "No business value specified"),
        "potential_insights": reasoning.get("potential_insights", "No insights specified")
    }


def generate_column_analysis_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate comprehensive summary of column analysis across all results.

    Args:
        results: List of all query results with column analysis

    Returns:
        Dict containing summary of column analysis insights
    """
    total_charts = len(results)
    successful_charts = len([r for r in results if r.get("success", False)])

    # Collect all column insights
    all_column_insights = []
    business_domains = set()
    chart_types = set()
    data_quality_issues = []
    confidence_scores = []

    for result in results:
        if result.get("success") and result.get("column_insights"):
            insights = result["column_insights"]
            all_column_insights.append(insights)

            # Collect metrics
            business_domains.add(insights.get("business_domain", "unknown"))
            chart_types.add(insights.get("chart_type", "unknown"))
            confidence_scores.append(insights.get("confidence_score", 0))

            # Collect data quality issues
            if insights.get("data_quality_issues", 0) > 0:
                data_quality_issues.append({
                    "section": result.get("section_name", "Unknown"),
                    "query": result.get("query", "Unknown"),
                    "issues_count": insights.get("data_quality_issues", 0)
                })

    # Calculate averages
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Generate recommendations
    recommendations = generate_overall_recommendations(all_column_insights, data_quality_issues)

    return {
        "total_charts_analyzed": successful_charts,
        "business_domains_covered": list(business_domains),
        "chart_types_used": list(chart_types),
        "average_confidence_score": round(avg_confidence, 2),
        "data_quality_issues_found": len(data_quality_issues),
        "overall_recommendations": recommendations,
        "column_analysis_coverage": {
            "charts_with_column_analysis": successful_charts,
            "percentage_coverage": (successful_charts / total_charts * 100) if total_charts > 0 else 0
        }
    }


def generate_overall_recommendations(column_insights: List[Dict[str, Any]], data_quality_issues: List[Dict[str, Any]]) -> List[str]:
    """
    Generate overall recommendations based on column analysis.

    Args:
        column_insights: List of column insights from all charts
        data_quality_issues: List of data quality issues found

    Returns:
        List of overall recommendations
    """
    recommendations = []

    # Analyze chart type diversity
    chart_types = [insight.get("chart_type", "unknown") for insight in column_insights]
    unique_chart_types = len(set(chart_types))

    if unique_chart_types < 3:
        recommendations.append("Consider using more diverse chart types for better data representation")

    # Analyze confidence scores
    confidence_scores = [insight.get("confidence_score", 0) for insight in column_insights]
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    if avg_confidence < 7:
        recommendations.append("Overall confidence is low - consider data quality improvements")

    # Analyze data quality issues
    if data_quality_issues:
        recommendations.append(f"Found {len(data_quality_issues)} data quality issues - prioritize data cleaning")

    # Analyze business domain coverage
    domains = [insight.get("business_domain", "unknown") for insight in column_insights]
    unique_domains = len(set(domains))

    if unique_domains > 1:
        recommendations.append(f"Analysis covers {unique_domains} business domains - ensure consistent terminology")

    return recommendations

# HTML generation function removed - no longer needed
# The system now returns all data directly without pagination or HTML generation

def get_all_results_as_json_dict(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    use_optimized: bool = True
) -> Dict[str, Any]:
    """
    Get all results as a JSON-serializable dictionary.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        use_optimized (bool): Whether to use optimized processing

    Returns:
        Dict[str, Any]: JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting all results as JSON dictionary from database ID {db_id}")

        # Choose the appropriate function based on optimization preference
        if use_optimized:
            print("🎯 Using optimized processing...")
            result = process_all_queries_with_graph_generation_optimized_sync(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )
        else:
            print("🔄 Using regular processing...")
            result = process_all_queries_with_graph_generation_sync(
                db_id=db_id,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            )

        # Convert the result to a JSON-serializable format
        json_result = serialize_for_json(result)

        print(f"✅ Successfully converted results to JSON format")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries processed: {json_result.get('total_queries', 0)}")
        print(f"   Successful: {json_result.get('successful_queries', 0)}")
        print(f"   Failed: {json_result.get('failed_queries', 0)}")

        return json_result

    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def get_results_as_json_simple(
    db_id: int = 1,
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Simple function to get results as JSON immediately.
    This function handles JSON serialization issues directly.

    Args:
        db_id (int): Database ID to fetch report structure from (default: 1)
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Clean JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting results as JSON for user: {user_id} from database ID: {db_id}")

        # Get the results using optimized processing
        result = process_all_queries_with_graph_generation_optimized_sync(
            db_id=db_id,
            user_id=user_id,
            export_format=export_format,
            theme=theme,
            analysis_subject=analysis_subject
        )

        print("✅ Results obtained successfully")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries: {result.get('total_queries', 0)}")
        print(f"   Successful: {result.get('successful_queries', 0)}")
        print(f"   Failed: {result.get('failed_queries', 0)}")

        # Use our enhanced serializer to handle all types properly
        clean_result = serialize_for_json(result)

        print(f"✅ Successfully converted to JSON format")
        print(f"   Database ID: {db_id}")
        print(f"   Total queries processed: {clean_result.get('total_queries', 0)}")
        print(f"   Successful: {clean_result.get('successful_queries', 0)}")
        print(f"   Failed: {clean_result.get('failed_queries', 0)}")

        return clean_result

    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "database_id": db_id,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

# Keep the old functions for backward compatibility
def get_all_results_as_json_dict_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string",
    use_optimized: bool = True
) -> Dict[str, Any]:
    """
    Get all results as a JSON-serializable dictionary from file.

    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis
        use_optimized (bool): Whether to use optimized processing

    Returns:
        Dict[str, Any]: JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting all results as JSON dictionary from {file_path}")

        # Choose the appropriate function based on optimization preference
        if use_optimized:
            print("🎯 Using optimized processing...")
            result = asyncio.run(process_all_queries_with_graph_generation_optimized_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            ))
        else:
            print("🔄 Using regular processing...")
            result = asyncio.run(process_all_queries_with_graph_generation_from_file(
                file_path=file_path,
                user_id=user_id,
                export_format=export_format,
                theme=theme,
                analysis_subject=analysis_subject
            ))

        # Convert the result to a JSON-serializable format
        json_result = serialize_for_json(result)

        print(f"✅ Successfully converted results to JSON format")
        print(f"   File: {file_path}")
        print(f"   Total queries processed: {json_result.get('total_queries', 0)}")
        print(f"   Successful: {json_result.get('successful_queries', 0)}")
        print(f"   Failed: {json_result.get('failed_queries', 0)}")

        return json_result

    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def get_results_as_json_simple_from_file(
    file_path: str = "Report_generator/utilites/testing.txt",
    user_id: str = "nabil",
    export_format: str = "png",
    theme: str = "modern",
    analysis_subject: str = "string"
) -> Dict[str, Any]:
    """
    Simple function to get results as JSON immediately from file.

    Args:
        file_path (str): Path to the testing.txt file
        user_id (str): User ID for the requests
        export_format (str): Export format for graphs (png, svg, pdf)
        theme (str): Theme for graphs (modern, dark, light, colorful)
        analysis_subject (str): Subject for data analysis

    Returns:
        Dict[str, Any]: Clean JSON-serializable dictionary with all results
    """
    try:
        print(f"🚀 Getting results as JSON for user: {user_id} from file: {file_path}")

        # Get the results using optimized processing
        result = asyncio.run(process_all_queries_with_graph_generation_optimized_from_file(
            file_path=file_path,
            user_id=user_id,
            export_format=export_format,
            theme=theme,
            analysis_subject=analysis_subject
        ))

        print("✅ Results obtained successfully")
        print(f"   File: {file_path}")
        print(f"   Total queries: {result.get('total_queries', 0)}")
        print(f"   Successful: {result.get('successful_queries', 0)}")
        print(f"   Failed: {result.get('failed_queries', 0)}")

        # Use our enhanced serializer to handle all types properly
        clean_result = serialize_for_json(result)

        print(f"✅ Successfully converted to JSON format")
        print(f"   File: {file_path}")
        print(f"   Total queries processed: {clean_result.get('total_queries', 0)}")
        print(f"   Successful: {clean_result.get('successful_queries', 0)}")
        print(f"   Failed: {clean_result.get('failed_queries', 0)}")

        return clean_result

    except Exception as e:
        print(f"❌ Error getting results as JSON: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {}
        }

def convert_result_to_json_serializable(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a result dictionary to be JSON serializable by handling date objects and other non-serializable types.
    """
    try:
        # Use our enhanced serializer to handle all types properly
        return serialize_for_json(result)
    except Exception as e:
        print(f"Error: Could not serialize result: {e}")
        import traceback
        traceback.print_exc()

        # Return a basic error response
        return {
            "success": False,
            "error": f"Serialization failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "total_queries": result.get("total_queries", 0),
            "successful_queries": 0,
            "failed_queries": result.get("total_queries", 0),
            "results": [],
            "summary": {"error": "Failed to serialize results"}
        }

def generate_complete_report(user_id: str, user_query: str = None, task_id: str = None) -> Dict[str, Any]:
    """
    MAIN FUNCTION: Generate a complete report for a user with user_id and optional user query.

    This function:
    1. Gets the database ID from user_id automatically
    2. Fetches and parses the report structure
    3. If user_query is provided, customizes queries based on user's specific request (extracts ALL queries)
    4. Generates graphs as images using optimized batch processing
    5. Returns the complete report with all data directly (no pagination, no HTML generation)

    Args:
        user_id (str): User ID to generate report for
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary", "salary analysis for July 2024")
        task_id (str, optional): Task ID for tracking current query processing

    Returns:
        Dict[str, Any]: Complete report with success status, results, graphs as images, and summary.
                       All data is returned directly without pagination or HTML generation.
    """
    try:
        print(f"🚀 Generating complete report for user: {user_id}")
        start_time = time.time()

        # Step 1: Get the database ID for the user using direct database manager call
        print(f"📊 Getting database ID for user: {user_id}")

        if DIRECT_DB_AVAILABLE:
            print("🎯 Using direct database manager call")
            try:
                user_data = db_manager.get_user_current_db_details(user_id)
                if not user_data or not user_data.get('db_id'):
                    return {
                        "success": False,
                        "error": f"No current database found for user {user_id}. Please set a current database first.",
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "total_queries": 0,
                        "successful_queries": 0,
                        "failed_queries": 0,
                        "results": [],
                        "summary": {}
                    }

                db_id = user_data['db_id']
                print(f"✅ Found database ID {db_id} for user {user_id}")

            except Exception as e:
                print(f"❌ Error getting database ID: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "results": [],
                    "summary": {}
                }
        else:
            # Fallback to sync version if direct database is not available
            print("🔄 Using sync database ID lookup")
            try:
                db_id = cached_get_db_id_from_user_id_sync(user_id)
                print(f"✅ Found database ID {db_id} for user {user_id}")
            except Exception as e:
                print(f"❌ Error getting database ID: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": 0,
                    "successful_queries": 0,
                    "failed_queries": 0,
                    "results": [],
                    "summary": {}
                }

        # Step 2: Generate the complete report using optimized batch processing
        if user_query:
            print(f"📈 Generating targeted report for database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"📈 Generating complete report for database ID {db_id}")

        # Use the optimized processing function with PNG export for images
        result = process_all_queries_with_graph_generation_optimized_sync(
            db_id=db_id,
            user_id=user_id,
            export_format="png",  # Always use PNG for images
            theme="modern",       # Use modern theme
            analysis_subject="data analysis",
            user_query=user_query,  # Pass user query for filtering
            task_id=task_id  # Pass task_id for current query tracking
        )

        total_time = time.time() - start_time

        # Step 3: Enhance the result with additional metadata
        if result.get("success"):
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()

            # Update summary with user info
            if "summary" in result:
                result["summary"]["user_id"] = user_id
                result["summary"]["generation_time"] = total_time
                result["summary"]["timestamp"] = datetime.now().isoformat()

            print(f"🎉 Report generation completed successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Database ID: {db_id}")
            print(f"   Total queries: {result.get('total_queries', 0)}")
            print(f"   Successful: {result.get('successful_queries', 0)}")
            print(f"   Failed: {result.get('failed_queries', 0)}")
            print(f"   Total generation time: {total_time:.2f}s")
            print(f"📈 Success Rate: {result.get('summary', {}).get('success_rate', 0):.1f}%")
        else:
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            print(f"❌ Report generation failed for user {user_id}")

        # Convert to JSON-serializable format
        return serialize_for_json(result)

    except Exception as e:
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Critical error in report generation for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": f"Critical error in report generation: {str(e)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "generation_time": total_time,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {"error": "Critical failure in report generation"}
        }

async def generate_complete_report_async(user_id: str, user_query: str = None, task_id: str = None) -> Dict[str, Any]:
    """
    Async version of generate_complete_report.

    Args:
        user_id (str): User ID to generate report for
        user_query (str, optional): User's specific request (e.g., "financial report", "attendance summary", "salary analysis for July 2024")
        task_id (str, optional): Task ID for tracking current query processing

    Returns:
        Dict[str, Any]: Complete report with success status, results, graphs as images, and summary.
                       All data is returned directly without pagination or HTML generation.
    """
    try:
        print(f"🚀 Generating complete report (async) for user: {user_id}")
        start_time = time.time()

        # Step 1: Get the database ID for the user
        print(f"📊 Getting database ID for user: {user_id}")

        try:
            db_id = await get_db_id_from_user_id(user_id)
            print(f"✅ Found database ID {db_id} for user {user_id}")
        except Exception as e:
            print(f"❌ Error getting database ID: {e}")
            return {
                "success": False,
                "error": f"Failed to get database ID for user {user_id}: {str(e)}",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "results": [],
                "summary": {}
            }

        # Step 2: Generate the complete report using optimized batch processing
        if user_query:
            print(f"📈 Generating targeted report for database ID {db_id}")
            print(f"🎯 User Query: {user_query}")
        else:
            print(f"📈 Generating complete report for database ID {db_id}")

        # Use the optimized async processing function with PNG export for images
        result = await process_all_queries_with_graph_generation_optimized(
            db_id=db_id,
            user_id=user_id,
            export_format="png",  # Always use PNG for images
            theme="modern",       # Use modern theme
            analysis_subject="data analysis",
            user_query=user_query,  # Pass user query for filtering
            task_id=task_id  # Pass task_id for current query tracking
        )

        total_time = time.time() - start_time

        # Step 3: Enhance the result with additional metadata
        if result.get("success"):
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()

            # Update summary with user info
            if "summary" in result:
                result["summary"]["user_id"] = user_id
                result["summary"]["generation_time"] = total_time
                result["summary"]["timestamp"] = datetime.now().isoformat()

            print(f"🎉 Async report generation completed successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Database ID: {db_id}")
            print(f"   Total queries: {result.get('total_queries', 0)}")
            print(f"   Successful: {result.get('successful_queries', 0)}")
            print(f"   Failed: {result.get('failed_queries', 0)}")
            print(f"   Total generation time: {total_time:.2f}s")
            print(f"📈 Success Rate: {result.get('summary', {}).get('success_rate', 0):.1f}%")
        else:
            result["user_id"] = user_id
            result["generation_time"] = total_time
            result["timestamp"] = datetime.now().isoformat()
            print(f"❌ Async report generation failed for user {user_id}")

        # Convert to JSON-serializable format
        return serialize_for_json(result)

    except Exception as e:
        total_time = time.time() - start_time if 'start_time' in locals() else 0
        print(f"❌ Critical error in async report generation for user {user_id}: {e}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": f"Critical error in async report generation: {str(e)}",
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "generation_time": total_time,
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "results": [],
            "summary": {"error": "Critical failure in async report generation"}
        }

def save_results_to_json_file(
    results: Dict[str, Any],
    output_file: str = "report_results.json"
) -> str:
    """
    Save results to a JSON file.

    Args:
        results: The results dictionary
        output_file: Output file path

    Returns:
        str: Path to the saved file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, default=serialize_for_json, indent=2, ensure_ascii=False)

        print(f"✅ Results saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"❌ Error saving results to file: {e}")
        return ""

if __name__ == "__main__":
    try:
        print("="*80)
        print("🚀 OPTIMIZED REPORT GENERATION SYSTEM")
        print("="*80)
        print("📝 This system now requires only user_id as input!")
        print("📊 Database ID is automatically fetched from the user's current database setting")
        print("🎯 Graphs are generated as PNG images with modern theme")
        print("📈 Uses optimized batch processing for maximum performance")
        print("📋 Returns all data directly without pagination or HTML generation")
        print("⚡ Features connection pooling and caching for better performance")
        print("="*80)

        # ✅ Set your inputs here (no need to type in terminal)
        test_user_id = "nabil"              # Default user_id
        user_query = "financial report 2023"  # Or None if you want the full report

        print(f"📋 Using user_id: {test_user_id}")
        if user_query:
            print(f"🎯 Targeted query: {user_query}")
        else:
            print("💡 No specific query provided - generating complete report")

        print("-"*80)

        # Generate the report
        start_time = time.time()
        report_result = generate_complete_report(test_user_id, user_query)
        total_time = time.time() - start_time

        print("-"*80)
        print("📊 FINAL REPORT SUMMARY:")
        print("-"*80)

        if report_result.get("success"):
            print(f"✅ SUCCESS: Report generated successfully!")
            print(f"👤 User ID: {report_result.get('user_id', 'Unknown')}")
            print(f"🗄️ Database ID: {report_result.get('database_id', 'Unknown')}")
            print(f"📝 Total Queries: {report_result.get('total_queries', 0)}")
            print(f"✅ Successful: {report_result.get('successful_queries', 0)}")
            print(f"❌ Failed: {report_result.get('failed_queries', 0)}")
            print(f"⏱️ Generation Time: {total_time:.2f} seconds")
            print(f"📈 Success Rate: {report_result.get('summary', {}).get('success_rate', 0):.1f}%")

            sections_processed = report_result.get('summary', {}).get('sections_processed', 0)
            total_sections = report_result.get('summary', {}).get('total_sections', 0)
            print(f"📂 Sections Processed: {sections_processed}/{total_sections}")

            method = report_result.get('summary', {}).get('processing_method', 'Unknown')
            print(f"⚙️ Processing Method: {method}")

            errors_summary = report_result.get('summary', {}).get('errors_summary', {})
            if errors_summary:
                print(f"⚠️ Errors Summary:")
                for error_type, count in errors_summary.items():
                    print(f"   - {error_type}: {count} occurrences")

            print(f"\n🎉 Report generation completed successfully!")
            print(f"📊 All data returned directly in JSON format")

            # Save results to file
            print(f"\n💾 Saving results to JSON file...")
            timestamp = int(time.time())
            output_file = f"report_results_{test_user_id}_{timestamp}.json"
            saved_file = save_results_to_json_file(report_result, output_file)
            if saved_file:
                print(f"✅ Results saved to: {saved_file}")
            else:
                print(f"❌ Failed to save results to file")

        else:
            print(f"❌ FAILED: Report generation failed")
            print(f"👤 User ID: {report_result.get('user_id', 'Unknown')}")
            print(f"🚨 Error: {report_result.get('error', 'Unknown error')}")
            print(f"⏱️ Time Before Failure: {total_time:.2f} seconds")

        print("="*80)
        print("🔚 Optimized report generation process completed")
        print("="*80)

    except KeyboardInterrupt:
        print(f"\n🛑 Process interrupted by user")
        # Cleanup connection pool
        db_pool.close_all()
        exit(0)
    except Exception as e:
        print(f"\n💥 Critical error in main function: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup connection pool
        db_pool.close_all()
        exit(1)
    finally:
        # Always cleanup connection pool
        db_pool.close_all()