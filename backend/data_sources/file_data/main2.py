from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import asyncio
from fastapi.concurrency import run_in_threadpool
import os
import shutil
import tempfile
from pathlib import Path
import uuid
from datetime import datetime, timedelta
from enum import Enum
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import json
from contextlib import contextmanager
from app.semi_structured.data_upload_v2 import process_file_pipeline
# Add these imports at the top of your file
from enum import Enum

# SQLite Database Manager
class DatabaseManager:
    def __init__(self, db_path: str = "task_tracking.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Create tasks table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    progress TEXT DEFAULT 'Pending',
                    current_file TEXT,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    error TEXT,
                    results TEXT  -- JSON string
                )
            ''')
            
            # Create bundles table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bundles (
                    bundle_id TEXT PRIMARY KEY,
                    task_ids TEXT NOT NULL,  -- JSON array of task IDs
                    total_files INTEGER NOT NULL,
                    filenames TEXT NOT NULL,  -- JSON array of filenames
                    created_at TEXT NOT NULL,
                    completed_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    current_processing_files TEXT DEFAULT '[]',  -- JSON array
                    remaining_files TEXT NOT NULL,  -- JSON array
                    status TEXT DEFAULT 'PENDING',
                    progress_percentage REAL DEFAULT 0.0,
                    last_updated TEXT NOT NULL
                )
            ''')
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Task operations
    def create_task(self, task_info: 'TaskInfo'):
        """Create a new task in the database."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO tasks (
                    task_id, status, created_at, started_at, completed_at,
                    progress, current_file, total_files, processed_files,
                    processing_time_seconds, error, results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_info.task_id,
                task_info.status.value,
                task_info.created_at.isoformat(),
                task_info.started_at.isoformat() if task_info.started_at else None,
                task_info.completed_at.isoformat() if task_info.completed_at else None,
                task_info.progress,
                task_info.current_file,
                task_info.total_files,
                task_info.processed_files,
                task_info.processing_time_seconds,
                task_info.error,
                json.dumps(task_info.results) if task_info.results else None
            ))
            conn.commit()
    
    def get_task(self, task_id: str) -> Optional['TaskInfo']:
        """Get a task by ID."""
        with self.get_connection() as conn:
            row = conn.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,)).fetchone()
            if row:
                return self._row_to_task_info(row)
            return None
    
    def update_task(self, task_id: str, **kwargs):
        """Update task fields."""
        if not kwargs:
            return
        
        # Convert datetime objects to ISO format
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                kwargs[key] = value.isoformat()
            elif isinstance(value, TaskStatus):
                kwargs[key] = value.value
            elif key == 'results' and value is not None:
                kwargs[key] = json.dumps(value)
        
        set_clause = ', '.join(f'{key} = ?' for key in kwargs.keys())
        query = f'UPDATE tasks SET {set_clause} WHERE task_id = ?'
        
        with self.get_connection() as conn:
            conn.execute(query, list(kwargs.values()) + [task_id])
            conn.commit()
    
    def get_all_tasks(self) -> List['TaskInfo']:
        """Get all tasks."""
        with self.get_connection() as conn:
            rows = conn.execute('SELECT * FROM tasks ORDER BY created_at DESC').fetchall()
            return [self._row_to_task_info(row) for row in rows]
    
    def _row_to_task_info(self, row) -> 'TaskInfo':
        """Convert database row to TaskInfo object."""
        return TaskInfo(
            task_id=row['task_id'],
            status=TaskStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            progress=row['progress'],
            current_file=row['current_file'],
            total_files=row['total_files'],
            processed_files=row['processed_files'],
            processing_time_seconds=row['processing_time_seconds'],
            error=row['error'],
            results=json.loads(row['results']) if row['results'] else None
        )
    
    # Bundle operations
    def create_bundle(self, bundle_info: 'BundleTaskInfo'):
        """Create a new bundle in the database."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO bundles (
                    bundle_id, task_ids, total_files, filenames, created_at,
                    completed_files, failed_files, current_processing_files,
                    remaining_files, status, progress_percentage, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bundle_info.bundle_id,
                json.dumps(bundle_info.task_ids),
                bundle_info.total_files,
                json.dumps(bundle_info.filenames),
                bundle_info.created_at.isoformat(),
                bundle_info.completed_files,
                bundle_info.failed_files,
                json.dumps(bundle_info.current_processing_files),
                json.dumps(bundle_info.remaining_files),
                bundle_info.status,
                bundle_info.progress_percentage,
                bundle_info.last_updated.isoformat()
            ))
            conn.commit()
    
    def get_bundle(self, bundle_id: str) -> Optional['BundleTaskInfo']:
        """Get a bundle by ID."""
        with self.get_connection() as conn:
            row = conn.execute('SELECT * FROM bundles WHERE bundle_id = ?', (bundle_id,)).fetchone()
            if row:
                return self._row_to_bundle_info(row)
            return None
    
    def update_bundle(self, bundle_id: str, **kwargs):
        """Update bundle fields."""
        if not kwargs:
            return
        
        # Convert datetime objects and lists to proper format
        for key, value in kwargs.items():
            if isinstance(value, datetime):
                kwargs[key] = value.isoformat()
            elif isinstance(value, list):
                kwargs[key] = json.dumps(value)
        
        set_clause = ', '.join(f'{key} = ?' for key in kwargs.keys())
        query = f'UPDATE bundles SET {set_clause} WHERE bundle_id = ?'
        
        with self.get_connection() as conn:
            conn.execute(query, list(kwargs.values()) + [bundle_id])
            conn.commit()
    
    def get_all_bundles(self) -> List['BundleTaskInfo']:
        """Get all bundles."""
        with self.get_connection() as conn:
            rows = conn.execute('SELECT * FROM bundles ORDER BY created_at DESC').fetchall()
            return [self._row_to_bundle_info(row) for row in rows]
    
    def _row_to_bundle_info(self, row) -> 'BundleTaskInfo':
        """Convert database row to BundleTaskInfo object."""
        bundle = BundleTaskInfo(
            bundle_id=row['bundle_id'],
            task_ids=json.loads(row['task_ids']),
            total_files=row['total_files'],
            filenames=json.loads(row['filenames']),
            created_at=datetime.fromisoformat(row['created_at'])
        )
        bundle.completed_files = row['completed_files']
        bundle.failed_files = row['failed_files']
        bundle.current_processing_files = json.loads(row['current_processing_files'])
        bundle.remaining_files = json.loads(row['remaining_files'])
        bundle.status = row['status']
        bundle.progress_percentage = row['progress_percentage']
        bundle.last_updated = datetime.fromisoformat(row['last_updated'])
        return bundle
    
    def cleanup_old_tasks(self, older_than_hours: int = 24):
        """Remove old completed tasks."""
        cutoff_time = (datetime.now() - timedelta(hours=older_than_hours)).isoformat()
        with self.get_connection() as conn:
            # Count tasks to be deleted
            count = conn.execute('''
                SELECT COUNT(*) FROM tasks 
                WHERE status IN ('completed', 'failed') 
                AND created_at < ?
            ''', (cutoff_time,)).fetchone()[0]
            
            # Delete old tasks
            conn.execute('''
                DELETE FROM tasks 
                WHERE status IN ('completed', 'failed') 
                AND created_at < ?
            ''', (cutoff_time,))
            conn.commit()
            return count
    
    def cleanup_old_bundles(self, older_than_hours: int = 24):
        """Remove old completed bundles."""
        cutoff_time = (datetime.now() - timedelta(hours=older_than_hours)).isoformat()
        with self.get_connection() as conn:
            # Count bundles to be deleted
            count = conn.execute('''
                SELECT COUNT(*) FROM bundles 
                WHERE status IN ('COMPLETED', 'FAILED') 
                AND last_updated < ?
            ''', (cutoff_time,)).fetchone()[0]
            
            # Delete old bundles
            conn.execute('''
                DELETE FROM bundles 
                WHERE status IN ('COMPLETED', 'FAILED') 
                AND last_updated < ?
            ''', (cutoff_time,))
            conn.commit()
            return count

# Initialize database manager
db_manager = DatabaseManager()

# Add these enums after your existing imports
class ChunkSourceType(Enum):
    """Types of chunk sources to use for answer generation."""
    NORMAL = "normal"
    FILTERED = "filtered"
    OVERLAP = "overlap"
    RERANKED = "reranked"

class AnswerStyle(Enum):
    """Different answer generation styles."""
    DETAILED = "detailed"
    CONCISE = "concise"
    BULLET_POINTS = "bullet_points"
    ANALYTICAL = "analytical"

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Add these new request/response models after your existing models
class EnhancedSearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    use_intent_reranker: Optional[bool] = False
    use_chunk_reranker: Optional[bool] = False
    use_dual_embeddings: Optional[bool] = True
    intent_top_k: Optional[int] = 20
    chunk_top_k: Optional[int] = 40
    chunk_source: Optional[ChunkSourceType] = ChunkSourceType.RERANKED
    max_chunks_for_answer: Optional[int] = 40
    answer_style: Optional[AnswerStyle] = AnswerStyle.DETAILED

class AnswerResults(BaseModel):
    answer: str
    sources_used: int
    confidence: str

class EnhancedSearchResponse(BaseModel):
    query: str
    answer: AnswerResults
    status: str
    configuration: Dict
    metadata: Optional[Dict] = None

class TaskInfo(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: str = "Pending"
    current_file: Optional[str] = None
    total_files: int = 0
    processed_files: int = 0
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    results: Optional[Dict] = None

class ProcessTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class FileProcessingResult(BaseModel):
    filename: str
    chunks: int
    processing_time_seconds: float

class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    total_files: int
    successful_files: int
    failed_files: int
    total_chunks: int
    total_processing_time_seconds: float
    file_results: List[FileProcessingResult]
    error: Optional[str] = None

# Import your pipeline functions
from app.unstructured.agent.data_upload_v2 import (
    process_single_document,
    run_store_only,
    run_embed_only,
    run_sub_intent_only,
    run_intent_mapping_only
)

app = FastAPI(title="Document Processing API")

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global task tracking
# task_store: Dict[str, TaskInfo] = {}
# task_lock = threading.Lock()

# Bundle storage (using database)
# bundle_store: Dict[str, BundleTaskInfo] = {}
# bundle_lock = Lock()

# Helper function to extract clean filename
def extract_clean_filename(stored_filename: str) -> str:
    """
    Extract clean filename from stored filename.
    Example: '1. Summary by Ministry  Division_20250710_183705_4bab0536.pdf' 
    Returns: 'Summary by Ministry  Division'
    """
    # Remove the file extension first
    name_without_ext = Path(stored_filename).stem
    
    # Pattern to match: anything before the pattern _YYYYMMDD_HHMMSS_uniqueid
    pattern = r'^(.*?)_\d{8}_\d{6}_[a-f0-9]{8}$'
    match = re.match(pattern, name_without_ext)
    
    if match:
        clean_name = match.group(1)
        # Handle numbered prefixes like "1. "
        clean_name = re.sub(r'^\d+\.\s*', '', clean_name)
        return clean_name.strip()
    
    return name_without_ext

# Helper function to parse chunk report and clean file
def parse_and_clean_chunk_report(result_file: str = "./processing_results/document_chunk_numbers.txt") -> List[FileProcessingResult]:
    """
    Parse the chunk report file and return file processing results.
    Then clean the file content.
    """
    results = []
    total_chunks = 0
    
    if not os.path.exists(result_file):
        return results
    
    try:
        with open(result_file, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if "|Chunks:" in line:
                try:
                    # Parse: "uploads/filename.pdf |Chunks: N"
                    parts = line.split("|Chunks:")
                    file_path = parts[0].strip()
                    chunk_count = int(parts[1].strip())
                    
                    # Extract filename from path
                    stored_filename = Path(file_path).name
                    clean_filename = extract_clean_filename(stored_filename)
                    
                    # Calculate processing time (0.5s per chunk as per estimate_time endpoint)
                    processing_time = chunk_count * 0.5
                    
                    results.append(FileProcessingResult(
                        filename=clean_filename,
                        chunks=chunk_count,
                        processing_time_seconds=processing_time
                    ))
                    
                    total_chunks += chunk_count
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line '{line}': {e}")
                    continue
        
        # Clean the file content after parsing
        with open(result_file, "w") as f:
            f.write("")  # Clear the file
        
        return results
        
    except Exception as e:
        print(f"Error reading chunk report file: {e}")
        return results

# Helper function to save uploaded file with original filename preserved
async def save_uploaded_file(file: UploadFile, preserve_original_name: bool = True) -> tuple[str, str]:
    """
    Save uploaded file and return the file path and original filename.
    
    Args:
        file: The uploaded file
        preserve_original_name: If True, keeps original filename with unique suffix
                              If False, generates completely new filename
    
    Returns:
        tuple: (file_path, original_filename)
    """
    original_filename = file.filename or "unknown_file"
    
    if preserve_original_name and file.filename:
        # Extract name and extension
        file_path_obj = Path(file.filename)
        name_without_ext = file_path_obj.stem
        extension = file_path_obj.suffix
        
        # Create unique filename while preserving original structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # Use shorter UUID for readability
        
        # Format: originalname_timestamp_uniqueid.ext
        filename = f"{name_without_ext}_{timestamp}_{unique_id}{extension}"
    else:
        # Generate completely new filename (original behavior)
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(original_filename).suffix if original_filename else ""
        filename = f"{timestamp}_{file_id}{file_extension}"
    
    file_path = UPLOAD_DIR / filename
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return str(file_path), original_filename

# Helper function to cleanup file
def cleanup_file(file_path: str):
    """Remove file after processing."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not cleanup file {file_path}: {e}")

# --------------------
# Unstructured Pipeline Background Processing
# --------------------
from typing import List
from fastapi import File, Form, BackgroundTasks, UploadFile
import uuid
from datetime import datetime

 #Request model for unstructured pipeline
class BackgroundUnstructuredRequest(BaseModel):
    file_paths: List[str]  # List of file paths to process
    preserve_filenames: bool = True
    delay_between_files: float = 0.0
    
    # Store parameters
    max_pages_per_chunk: int = 5
    boundary_sentences: int = 3
    boundary_table_rows: int = 3
    target_pages_per_chunk: int = 3
    overlap_pages: int = 1
    min_pages_per_chunk: int = 1
    respect_boundaries: bool = True
    max_workers: int = 4
    
    # Embedding parameters
    embed_batch_size: int = 4
    embed_delay_between_requests: float = 5.0
    embed_max_retries: int = 3
    embed_retry_delay: float = 20.0
    
    # Sub-intent parameters
    sub_intent_similarity_threshold: float = 0.75
    sub_intent_batch_size: int = 4
    sub_intent_delay_between_requests: float = 5.0
    sub_intent_max_retries: int = 3
    sub_intent_retry_delay: float = 20.0
    
    # Intent mapping parameters
    intent_similarity_threshold: float = 0.75
    top_n_candidates: int = 5
    intent_batch_size: int = 20
    intent_delay: float = 1.0
    
    # New parameters
    file_description: Optional[str] = None
    table_name: Optional[str] = None
    user_id: Optional[str] = None

# Background task function for unstructured pipeline
async def run_complete_unstructured_pipeline_with_tracking(task_id: str, req: BackgroundUnstructuredRequest):
    """
    Background task that runs the complete unstructured pipeline with proper task tracking:
    1. Store pipeline (for each file)
    2. Embedding generation
    3. Sub-intent generation  
    4. Intent mapping
    """
    
    # Update task status to running
    # with task_lock: # This line is removed as per the new_code
    #     if task_id in task_store: # This line is removed as per the new_code
    #         task_store[task_id].status = TaskStatus.RUNNING # This line is removed as per the new_code
    #         task_store[task_id].started_at = datetime.now() # This line is removed as per the new_code
    #         task_store[task_id].progress = "Starting unstructured pipeline" # This line is removed as per the new_code
    
    # Use database manager
    task_info = db_manager.get_task(task_id)
    if task_info:
        db_manager.update_task(
            task_id, 
            status=TaskStatus.RUNNING.value, 
            started_at=datetime.now(), 
            progress="Starting unstructured pipeline"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    
    try:
        print("🚀 Starting complete unstructured background pipeline...")
        
        total_files = len(req.file_paths)
        print(f"📁 Processing {total_files} files")
        
        # Step 1: Process each file through store pipeline
        print(f"🔄 Step 1/4: Running store pipeline for {total_files} files...")
        
        processed_files = 0
        failed_files = []
        
        for i, file_path in enumerate(req.file_paths, 1):
            try:
                # Update progress
                db_manager.update_task(
                    task_id, 
                    progress=f"Step 1/4: Processing file structure", 
                    processed_files=processed_files,
                    current_file=Path(file_path).name
                )
                
                # Validate file exists
                if not os.path.exists(file_path):
                    print(f"❌ File not found: {file_path}")
                    failed_files.append(file_path)
                    continue
                
                print(f"📄 Processing file {i}/{total_files}: {Path(file_path).name}")
                
                # Run store pipeline for this file
                store_success = await asyncio.to_thread(
                    run_store_only,
                    file_path=file_path,
                    max_pages_per_chunk=req.max_pages_per_chunk,
                    boundary_sentences=req.boundary_sentences,
                    boundary_table_rows=req.boundary_table_rows,
                    target_pages_per_chunk=req.target_pages_per_chunk,
                    overlap_pages=req.overlap_pages,
                    min_pages_per_chunk=req.min_pages_per_chunk,
                    respect_boundaries=req.respect_boundaries,
                    max_workers=req.max_workers,
                    file_description=req.file_description,
                    table_name=req.table_name,
                    user_id=req.user_id
                )
                
                if store_success:
                    processed_files += 1
                    print(f"✅ File {i}/{total_files} processed successfully")
                else:
                    print(f"❌ File {i}/{total_files} failed to process")
                    failed_files.append(file_path)
                
                # Add delay between files if specified
                if req.delay_between_files > 0 and i < total_files:
                    await asyncio.sleep(req.delay_between_files)
                    
            except Exception as e:
                print(f"❌ Error processing file {file_path}: {str(e)}")
                failed_files.append(file_path)
        
        # Check if any files were processed successfully
        if processed_files == 0:
            error_msg = "No files were processed successfully"
            print(f"❌ {error_msg}")
            
            # Update task with error
            db_manager.update_task(
                task_id, 
                status=TaskStatus.FAILED.value, 
                completed_at=datetime.now(), 
                error=error_msg, 
                progress=f"Failed: {error_msg}"
            )
            
            return {"status": "error", "message": error_msg}
        
        print(f"✅ Store pipeline completed! Processed {processed_files}/{total_files} files")
        
        # Step 2: Run embedding generation
        print("🔄 Step 2/4: Running embedding generation...")
        # Use database manager
        db_manager.update_task(task_id, progress="Step 2/4: Generating embeddings")
        
        embed_result = await asyncio.to_thread(
            run_embed_only,
            batch_size=req.embed_batch_size,
            delay_between_requests=req.embed_delay_between_requests,
            max_retries=req.embed_max_retries,
            retry_delay=req.embed_retry_delay
        )
        print(f"✅ Embedding generation completed: {embed_result}")
        
        # Step 3: Run sub-intent generation
        print("🔄 Step 3/4: Running sub-intent generation...")
        # Use database manager
        db_manager.update_task(task_id, progress="Step 3/4: Generating sub-intents")
        
        sub_intent_result = await asyncio.to_thread(
            run_sub_intent_only,
            similarity_threshold=req.sub_intent_similarity_threshold,
            batch_size=req.sub_intent_batch_size,
            delay_between_requests=req.sub_intent_delay_between_requests,
            max_retries=req.sub_intent_max_retries,
            retry_delay=req.sub_intent_retry_delay
        )
        print(f"✅ Sub-intent generation completed: {sub_intent_result}")
        
        # Step 4: Run intent mapping
        print("🔄 Step 4/4: Running intent mapping...")
        # Use database manager
        db_manager.update_task(task_id, progress="Step 4/4: Mapping intents")
        
        intent_mapping_result = await asyncio.to_thread(
            run_intent_mapping_only,
            intent_similarity_threshold=req.intent_similarity_threshold,
            top_n_candidates=req.top_n_candidates,
            intent_batch_size=req.intent_batch_size,
            intent_delay=req.intent_delay
        )
        print(f"✅ Intent mapping completed: {intent_mapping_result}")
        
        print("🎉 Complete unstructured pipeline finished successfully!")
        
        # Prepare results
        results = {
            "status": "success",
            "message": "Complete unstructured pipeline finished successfully",
            "total_files": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files,
            "file_paths": req.file_paths,
            "results": {
                "embed_result": embed_result,
                "sub_intent_result": sub_intent_result,
                "intent_mapping_result": intent_mapping_result
            }
        }
        
                # Update task with success
        task_info = db_manager.get_task(task_id)
        if task_info:
            completed_at = datetime.now()
            processing_time = None
            if task_info.started_at:
                processing_time = (completed_at - task_info.started_at).total_seconds()
            
            db_manager.update_task(
                task_id, 
                status=TaskStatus.COMPLETED.value, 
                completed_at=completed_at, 
                progress="Completed successfully", 
                results=results, 
                processed_files=processed_files, 
                processing_time_seconds=processing_time
            )
        
        return results
        
    except Exception as e:
        error_msg = f"Complete unstructured pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Update task with error
        task_info = db_manager.get_task(task_id)
        if task_info:
            completed_at = datetime.now()
            processing_time = None
            if task_info.started_at:
                processing_time = (completed_at - task_info.started_at).total_seconds()
            
            db_manager.update_task(
                task_id, 
                status=TaskStatus.FAILED.value, 
                completed_at=completed_at, 
                error=error_msg, 
                progress=f"Failed: {error_msg}",
                processing_time_seconds=processing_time
            )
        
        return {"status": "error", "message": error_msg}


# Unstructured file system endpoint

# @app.post("/unstructured_file_system", response_model=ProcessTaskResponse, summary="Upload files and run complete unstructured pipeline in background")
async def unstructured_file_system_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple files to upload and process through unstructured pipeline"),
    preserve_filenames: bool = Form(True, description="Whether to preserve original filenames"),
    delay_between_files: float = Form(0.0, description="Delay in seconds between processing each file"),
    
    # Store parameters
    max_pages_per_chunk: int = Form(5),
    boundary_sentences: int = Form(3),
    boundary_table_rows: int = Form(3),
    target_pages_per_chunk: int = Form(3),
    overlap_pages: int = Form(1),
    min_pages_per_chunk: int = Form(1),
    respect_boundaries: bool = Form(True),
    max_workers: int = Form(4),
    
    # Embedding parameters
    embed_batch_size: int = Form(4),
    embed_delay_between_requests: float = Form(5.0),
    embed_max_retries: int = Form(3),
    embed_retry_delay: float = Form(20.0),
    
    # Sub-intent parameters
    sub_intent_similarity_threshold: float = Form(0.75),
    sub_intent_batch_size: int = Form(4),
    sub_intent_delay_between_requests: float = Form(5.0),
    sub_intent_max_retries: int = Form(3),
    sub_intent_retry_delay: float = Form(20.0),
    
    # Intent mapping parameters
    intent_similarity_threshold: float = Form(0.75),
    top_n_candidates: int = Form(5),
    intent_batch_size: int = Form(20),
    intent_delay: float = Form(1.0),
    
    # New parameters
    file_description: str = Form(None, description="Optional description for the files"),
    table_name: str = Form(None, description="Optional table name for the files"),
    user_id: str = Form(None, description="Optional user id for the files")
):
    """
    Upload multiple files and run the complete unstructured pipeline in background:
    1. Store pipeline (for each file)
    2. Embedding generation
    3. Sub-intent generation
    4. Intent mapping
    
    Returns immediately with task ID while processing continues in background.
    """
    try:
        # Generate a unique task ID for tracking
        task_id = str(uuid.uuid4())
        
        # Save all uploaded files
        file_paths = []
        original_filenames = []
        
        for file in files:
            file_path, original_filename = await save_uploaded_file(file, preserve_filenames)
            file_paths.append(file_path)
            original_filenames.append(original_filename)
        
        # Create task info and store it
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            total_files=len(files),
            processed_files=0,
            progress="Queued for unstructured processing",
            current_file=original_filenames[0] if original_filenames else None
        )
        
        # Store task info
        # with task_lock: # This line is removed as per the new_code
        #     task_store[task_id] = task_info # This line is removed as per the new_code
        
        # Use database manager
        db_manager.create_task(task_info)
        
        # Create request object with file paths included
        req = BackgroundUnstructuredRequest(
            file_paths=file_paths,
            preserve_filenames=preserve_filenames,
            delay_between_files=delay_between_files,
            max_pages_per_chunk=max_pages_per_chunk,
            boundary_sentences=boundary_sentences,
            boundary_table_rows=boundary_table_rows,
            target_pages_per_chunk=target_pages_per_chunk,
            overlap_pages=overlap_pages,
            min_pages_per_chunk=min_pages_per_chunk,
            respect_boundaries=respect_boundaries,
            max_workers=max_workers,
            embed_batch_size=embed_batch_size,
            embed_delay_between_requests=embed_delay_between_requests,
            embed_max_retries=embed_max_retries,
            embed_retry_delay=embed_retry_delay,
            sub_intent_similarity_threshold=sub_intent_similarity_threshold,
            sub_intent_batch_size=sub_intent_batch_size,
            sub_intent_delay_between_requests=sub_intent_delay_between_requests,
            sub_intent_max_retries=sub_intent_max_retries,
            sub_intent_retry_delay=sub_intent_retry_delay,
            intent_similarity_threshold=intent_similarity_threshold,
            top_n_candidates=top_n_candidates,
            intent_batch_size=intent_batch_size,
            intent_delay=intent_delay,
            file_description=file_description,
            table_name=table_name,
            user_id=user_id
        )
        
        # Clear chunk report before starting
        # parse_and_clean_chunk_report()
        
        # Add the background task with proper task tracking
        background_tasks.add_task(run_complete_unstructured_pipeline_with_tracking, task_id, req)
        
        return ProcessTaskResponse(
            task_id=task_id,
            status="accepted",
            message=f"Unstructured pipeline started for {len(files)} files. Use task_id to check progress."
        )
        
    except Exception as e:
        # Cleanup uploaded files if error occurs
        # if 'file_paths' in locals(): # This line is removed as per the new_code
        #     for file_path in file_paths: # This line is removed as per the new_code
        #         cleanup_file(file_path) # This line is removed as per the new_code
        raise HTTPException(status_code=500, detail=f"Failed to start unstructured pipeline: {str(e)}")


# --------------------
# Request Models (Updated)
# --------------------
class EmbedRequest(BaseModel):
    batch_size: int = 4
    delay_between_requests: float = 5.0
    max_retries: int = 3
    retry_delay: float = 20.0

class SubIntentRequest(BaseModel):
    similarity_threshold: float = 0.75
    batch_size: int = 4
    delay_between_requests: float = 5.0
    max_retries: int = 3
    retry_delay: float = 20.0

class IntentMappingRequest(BaseModel):
    intent_similarity_threshold: float = 0.75
    top_n_candidates: int = 5
    intent_batch_size: int = 20
    intent_delay: float = 1.0

# --------------------
# QA System Integration
# --------------------
from app.unstructured.agent.query_answer import DocumentQASystem

# Initialize QA system instance once at startup
qa_system = DocumentQASystem()

# Pydantic request schema
class QuestionRequest(BaseModel):
    question: str
    use_reranker: Optional[bool] = True
    max_context_chunks: Optional[int] = 10

# Pydantic response schema
class QAResponse(BaseModel):
    query: str
    answer: str
    status: str

# --------------------
# Enhanced Search Integration
# --------------------
from app.unstructured.agent.query_answerv2 import EnhancedSearchAnswerPipeline
from app.unstructured.agent.query_answerv2 import DatabaseConfig

# Initialize enhanced search pipeline
API_KEY = os.getenv("google_api_key")
if not API_KEY:
    print("❌ Warning: GOOGLE_API_KEY environment variable not set")
    enhanced_pipeline = None
else:
    db_config = DatabaseConfig()
    enhanced_pipeline = EnhancedSearchAnswerPipeline(db_config, API_KEY)


@app.post("/search")
async def search_endpoint(request: EnhancedSearchRequest):
    """Enhanced search and answer endpoint with configurable parameters."""
    
    if enhanced_pipeline is None:
        raise HTTPException(
            status_code=500, 
            detail="Enhanced search pipeline not initialized. Please set GOOGLE_API_KEY environment variable."
        )
    
    try:
        # Call the enhanced search pipeline
        results = await enhanced_pipeline.search_and_answer(
            query=request.query,
            use_intent_reranker=request.use_intent_reranker,
            use_chunk_reranker=request.use_chunk_reranker,
            use_dual_embeddings=request.use_dual_embeddings,
            intent_top_k=request.intent_top_k,
            chunk_top_k=request.chunk_top_k,
            chunk_source=request.chunk_source,
            max_chunks_for_answer=request.max_chunks_for_answer,
            answer_style=request.answer_style
        )
        
        return {
            "query": request.query,
            "answer": results["answer_results"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# 5. DELETE the old run_background_pipeline_endpoint function completely
import asyncio
import os
import sys
from typing import Optional
from fastapi import BackgroundTasks
from pydantic import BaseModel

# 1. Update the BackgroundPipelineRequest model to include file_path
class BackgroundPipelineRequest(BaseModel):
    file_path: str  # Add this field to store the file path
    preserve_layout_alignment_across_pages: bool = False
    skip_diagonal_text: bool = True
    output_tables_as_HTML: bool = False
    disable_image_extraction: bool = False
    spreadsheet_extract_sub_tables: bool = True
    result_type: str = "markdown"
    chunk_size: int = 20000
    chunk_overlap: int = 1000
    
    # Embedding parameters
    embed_batch_size: int = 32
    embed_delay_between_requests: float = 0.1
    embed_max_retries: int = 3
    embed_retry_delay: float = 1.0
    
    # Sub-intent parameters
    similarity_threshold: float = 0.8
    sub_intent_batch_size: int = 16
    sub_intent_delay_between_requests: float = 0.1
    sub_intent_max_retries: int = 3
    sub_intent_retry_delay: float = 1.0
    
    # Intent mapping parameters
    intent_similarity_threshold: float = 0.7
    top_n_candidates: int = 5
    intent_batch_size: int = 8
    intent_delay: float = 0.2
    
    # New parameters
    file_description: Optional[str] = None
    table_name: Optional[str] = None
    user_id: Optional[str] = None

# 1. Update the background pipeline function to accept and use task_id
async def run_complete_background_pipeline_with_tracking(task_id: str, req: BackgroundPipelineRequest):
    """
    Background task that runs the complete pipeline with proper task tracking:
    1. File processing pipeline
    2. Embedding generation
    3. Sub-intent generation  
    4. Intent mapping
    """
    
    # Update task status to running
    # with task_lock: # This line is removed as per the new_code
    #     if task_id in task_store: # This line is removed as per the new_code
    #         task_store[task_id].status = TaskStatus.RUNNING # This line is removed as per the new_code
    #         task_store[task_id].started_at = datetime.now() # This line is removed as per the new_code
    #         task_store[task_id].progress = "Starting semi-structured pipeline" # This line is removed as per the new_code
    
    # Use database manager
    task_info = db_manager.get_task(task_id)
    if task_info:
        db_manager.update_task(
            task_id, 
            status=TaskStatus.RUNNING.value, 
            started_at=datetime.now(), 
            progress="Starting semi-structured pipeline"
        )
    else:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    
    try:
        print("🚀 Starting complete background pipeline...")
        
        # Step 1: Use file path from request
        file_path = req.file_path
        
        # Validate file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found at {file_path}"
            print(f"❌ Error: {error_msg}")
            
            # Update task with error
            # with task_lock: # This line is removed as per the new_code
            #     if task_id in task_store: # This line is removed as per the new_code
            #         task_store[task_id].status = TaskStatus.FAILED # This line is removed as per the new_code
            #         task_store[task_id].completed_at = datetime.now() # This line is removed as per the new_code
            #         task_store[task_id].error = error_msg # This line is removed as per the new_code
            #         task_store[task_id].progress = f"Failed: {error_msg}" # This line is removed as per the new_code
            
            # Use database manager
            with db_manager.get_connection() as conn:
                task_info = db_manager.get_task(task_id)
                if task_info:
                    task_info.status = TaskStatus.FAILED
                    task_info.completed_at = datetime.now()
                    task_info.error = error_msg
                    task_info.progress = f"Failed: {error_msg}"
                    db_manager.update_task(task_id, status=task_info.status.value, completed_at=task_info.completed_at.isoformat(), error=task_info.error, progress=task_info.progress)
            
            # cleanup_file(file_path) # This line is removed as per the new_code
            return {"status": "error", "message": error_msg}
        
        print(f"📁 Processing file: {file_path}")
        print(f"📊 Parameters: chunk_size={req.chunk_size}, chunk_overlap={req.chunk_overlap}")
        
        # Step 2: Run file processing pipeline
        print("🔄 Step 1/4: Running file processing pipeline...")
        # with task_lock: # This line is removed as per the new_code
        #     if task_id in task_store: # This line is removed as per the new_code
        #         task_store[task_id].progress = "Step 1/4: Processing file structure" # This line is removed as per the new_code
        
        # Use database manager
        db_manager.update_task(task_id, progress="Step 1/4: Processing file structure")
        
        file_success = await asyncio.to_thread(
            process_file_pipeline,
            file_path=file_path,
            preserve_layout_alignment_across_pages=req.preserve_layout_alignment_across_pages,
            skip_diagonal_text=req.skip_diagonal_text,
            output_tables_as_HTML=req.output_tables_as_HTML,
            disable_image_extraction=req.disable_image_extraction,
            spreadsheet_extract_sub_tables=req.spreadsheet_extract_sub_tables,
            result_type=req.result_type,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            user_id=req.user_id,
            file_description=req.file_description,
            table_name=req.table_name
        )
        
        if not file_success:
            error_msg = "File processing pipeline failed"
            print(f"❌ {error_msg}")
            
            # Update task with error
            # with task_lock: # This line is removed as per the new_code
            #     if task_id in task_store: # This line is removed as per the new_code
            #         task_store[task_id].status = TaskStatus.FAILED # This line is removed as per the new_code
            #         task_store[task_id].completed_at = datetime.now() # This line is removed as per the new_code
            #         task_store[task_id].error = error_msg # This line is removed as per the new_code
            #         task_store[task_id].progress = f"Failed: {error_msg}" # This line is removed as per the new_code
            
            # Use database manager
            with db_manager.get_connection() as conn:
                task_info = db_manager.get_task(task_id)
                if task_info:
                    task_info.status = TaskStatus.FAILED
                    task_info.completed_at = datetime.now()
                    task_info.error = error_msg
                    task_info.progress = f"Failed: {error_msg}"
                    db_manager.update_task(task_id, status=task_info.status.value, completed_at=task_info.completed_at.isoformat(), error=task_info.error, progress=task_info.progress)
            
            # cleanup_file(file_path) # This line is removed as per the new_code
            return {"status": "error", "message": error_msg}
        
        print("✅ File processing completed successfully!")
        
        # Step 3: Run embedding generation
        print("🔄 Step 2/4: Running embedding generation...")
        # with task_lock: # This line is removed as per the new_code
        #     if task_id in task_store: # This line is removed as per the new_code
        #         task_store[task_id].progress = "Step 2/4: Generating embeddings" # This line is removed as per the new_code
        
        # Use database manager
        db_manager.update_task(task_id, progress="Step 2/4: Generating embeddings")
        
        embed_result = await asyncio.to_thread(
            run_embed_only,
            batch_size=req.embed_batch_size,
            delay_between_requests=req.embed_delay_between_requests,
            max_retries=req.embed_max_retries,
            retry_delay=req.embed_retry_delay,
            user_id=req.user_id
        )
        print(f"✅ Embedding generation completed: {embed_result}")
        
        # Step 4: Run sub-intent generation
        print("🔄 Step 3/4: Running sub-intent generation...")
        # with task_lock: # This line is removed as per the new_code
        #     if task_id in task_store: # This line is removed as per the new_code
        #         task_store[task_id].progress = "Step 3/4: Generating sub-intents" # This line is removed as per the new_code
        
        # Use database manager
        db_manager.update_task(task_id, progress="Step 3/4: Generating sub-intents")
        
        sub_intent_result = await asyncio.to_thread(
            run_sub_intent_only,
            similarity_threshold=req.similarity_threshold,
            batch_size=req.sub_intent_batch_size,
            delay_between_requests=req.sub_intent_delay_between_requests,
            max_retries=req.sub_intent_max_retries,
            retry_delay=req.sub_intent_retry_delay,
            user_id=req.user_id
        )
        print(f"✅ Sub-intent generation completed: {sub_intent_result}")
        
        # Step 5: Run intent mapping
        print("🔄 Step 4/4: Running intent mapping...")
        # with task_lock: # This line is removed as per the new_code
        #     if task_id in task_store: # This line is removed as per the new_code
        #         task_store[task_id].progress = "Step 4/4: Mapping intents" # This line is removed as per the new_code
        
        # Use database manager
        db_manager.update_task(task_id, progress="Step 4/4: Mapping intents")
        
        intent_mapping_result = await asyncio.to_thread(
            run_intent_mapping_only,
            intent_similarity_threshold=req.intent_similarity_threshold,
            top_n_candidates=req.top_n_candidates,
            intent_batch_size=req.intent_batch_size,
            intent_delay=req.intent_delay,
            user_id=req.user_id
        )
        print(f"✅ Intent mapping completed: {intent_mapping_result}")
        
        print("🎉 Complete pipeline finished successfully!")
        
        # Clean up the uploaded file after processing
        # cleanup_file(file_path) # This line is removed as per the new_code
        
        # Prepare results
        results = {
            "status": "success",
            "message": "Complete background pipeline finished successfully",
            "file_path": file_path,
            "file_processing_success": file_success,
            "results": {
                "embed_result": embed_result,
                "sub_intent_result": sub_intent_result,
                "intent_mapping_result": intent_mapping_result
            }
        }
        
        # Update task with success
        task_info = db_manager.get_task(task_id)
        if task_info:
            completed_at = datetime.now()
            processing_time = None
            if task_info.started_at:
                processing_time = (completed_at - task_info.started_at).total_seconds()
            
            db_manager.update_task(
                task_id, 
                status=TaskStatus.COMPLETED.value, 
                completed_at=completed_at, 
                progress="Completed successfully", 
                results=results, 
                processing_time_seconds=processing_time
            )
        
        return results
        
    except Exception as e:
        error_msg = f"Complete pipeline failed: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Update task with error
        task_info = db_manager.get_task(task_id)
        if task_info:
            completed_at = datetime.now()
            processing_time = None
            if task_info.started_at:
                processing_time = (completed_at - task_info.started_at).total_seconds()
            
            db_manager.update_task(
                task_id, 
                status=TaskStatus.FAILED.value, 
                completed_at=completed_at, 
                error=error_msg, 
                progress=f"Failed: {error_msg}",
                processing_time_seconds=processing_time
            )
        
        # Clean up the uploaded file in case of error
        # if 'file_path' in locals(): # This line is removed as per the new_code
        #     cleanup_file(file_path) # This line is removed as per the new_code
        return {"status": "error", "message": error_msg}


# # 2. Fixed semi_structured_file_system endpoint
# @app.post("/semi_structured_file_system", response_model=ProcessTaskResponse, summary="Upload file and run complete semi-structured pipeline in background")
async def semi_structured_file_system_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Single file to upload and process through semi-structured pipeline"),
    preserve_filename: bool = Form(True, description="Whether to preserve original filename"),
    # File processing parameters
    preserve_layout_alignment_across_pages: bool = Form(False),
    skip_diagonal_text: bool = Form(True),
    output_tables_as_HTML: bool = Form(False),
    disable_image_extraction: bool = Form(False),
    spreadsheet_extract_sub_tables: bool = Form(True),
    result_type: str = Form("markdown"),
    chunk_size: int = Form(20000),
    chunk_overlap: int = Form(1000),
    
    # Embedding parameters
    embed_batch_size: int = Form(32),
    embed_delay_between_requests: float = Form(0.1),
    embed_max_retries: int = Form(3),
    embed_retry_delay: float = Form(1.0),
    
    # Sub-intent parameters
    similarity_threshold: float = Form(0.8),
    sub_intent_batch_size: int = Form(16),
    sub_intent_delay_between_requests: float = Form(0.1),
    sub_intent_max_retries: int = Form(3),
    sub_intent_retry_delay: float = Form(1.0),

    # Intent mapping parameters
    intent_similarity_threshold: float = Form(0.7),
    top_n_candidates: int = Form(5),
    intent_batch_size: int = Form(8),
    intent_delay: float = Form(0.2),
    
    # New parameters
    file_description: Optional[str] = Form(None, description="Optional description for the file"),
    table_name: Optional[str] = Form(None, description="Optional table name for the file"),
    user_id: Optional[str] = Form(None, description="Optional user id for the file")
):
    """
    Upload a single file and run the complete semi-structured pipeline in background:
    1. File processing
    2. Embedding generation
    3. Sub-intent generation
    4. Intent mapping
    
    Returns immediately with task ID while processing continues in background.
    """
    try:
        # Generate a unique task ID for tracking
        task_id = str(uuid.uuid4())
        
        # Save uploaded file
        file_path, original_filename = await save_uploaded_file(file, preserve_filename)
        
        # Create task info and store it
        task_info = TaskInfo(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            total_files=1,
            progress="Queued for semi-structured processing",
            current_file=original_filename
        )
        
        # Store task info
        # with task_lock: # This line is removed as per the new_code
        #     task_store[task_id] = task_info # This line is removed as per the new_code
        
        # Use database manager
        db_manager.create_task(task_info)
        
        # Create request object with file_path included
        req = BackgroundPipelineRequest(
            file_path=file_path,
            preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
            skip_diagonal_text=skip_diagonal_text,
            output_tables_as_HTML=output_tables_as_HTML,
            disable_image_extraction=disable_image_extraction,
            spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
            result_type=result_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embed_batch_size=embed_batch_size,
            embed_delay_between_requests=embed_delay_between_requests,
            embed_max_retries=embed_max_retries,
            embed_retry_delay=embed_retry_delay,
            similarity_threshold=similarity_threshold,
            sub_intent_batch_size=sub_intent_batch_size,
            sub_intent_delay_between_requests=sub_intent_delay_between_requests,
            sub_intent_max_retries=sub_intent_max_retries,
            sub_intent_retry_delay=sub_intent_retry_delay,
            intent_similarity_threshold=intent_similarity_threshold,
            top_n_candidates=top_n_candidates,
            intent_batch_size=intent_batch_size,
            intent_delay=intent_delay,
            file_description=file_description,
            table_name=table_name,
            user_id=user_id
        )
        
        # Clear chunk report before starting
        # parse_and_clean_chunk_report()
        
        # Add the background task with proper task tracking
        background_tasks.add_task(run_complete_background_pipeline_with_tracking, task_id, req)
        
        return ProcessTaskResponse(
            task_id=task_id,
            status="accepted",
            message=f"Semi-structured pipeline started for {original_filename}. Use task_id to check progress."
        )
        
    except Exception as e:
        # Cleanup uploaded file if error occurs
        # if 'file_path' in locals(): # This line is removed as per the new_code
        #     cleanup_file(file_path) # This line is removed as per the new_code
        raise HTTPException(status_code=500, detail=f"Failed to start semi-structured pipeline: {str(e)}")

# Add helper functions at the top of the file, after imports
import json
from typing import List, Optional, Union

def parse_file_metadata_arrays(
    file_descriptions: Optional[str], 
    table_names: Optional[str], 
    user_ids: Optional[str],
    file_count: int
) -> List[tuple]:
    """
    Parse file_descriptions, table_names, user_ids (JSON arrays, comma-separated, or single strings) 
    and return list of (description, table_name, user_id) tuples mapped to files.
    """
    def parse_field(field: Optional[str]) -> List[Optional[str]]:
        if field:
            # Try JSON first
            try:
                parsed = json.loads(field)
                if isinstance(parsed, list):
                    return parsed
                else:
                    return [str(parsed)] * file_count
            except Exception:
                # Try comma-separated fallback
                if ',' in field:
                    items = [item.strip().strip('"\'') for item in field.split(',')]
                    return items
                else:
                    return [field] * file_count
        else:
            return [None] * file_count
    
    descriptions = parse_field(file_descriptions)
    names = parse_field(table_names)
    users = parse_field(user_ids)
    
    while len(descriptions) < file_count:
        descriptions.append(None)
    while len(names) < file_count:
        names.append(None)
    while len(users) < file_count:
        users.append(None)
    
    return [(descriptions[i], names[i], users[i]) for i in range(file_count)]

@app.post("/smart_file_system_backend", response_model=Dict, summary="Upload multiple files and automatically route to appropriate processing pipeline")
async def smart_file_system_backend(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple files to upload and process automatically"),
    preserve_filenames: bool = Form(True, description="Whether to preserve original filenames"),
    
    # Common parameters for both pipelines
    delay_between_files: float = Form(0.2, description="Delay in seconds between processing each file"),
    
    # Unstructured pipeline parameters
    max_pages_per_chunk: int = Form(10),
    boundary_sentences: int = Form(3),
    boundary_table_rows: int = Form(3),
    target_pages_per_chunk: int = Form(3),
    overlap_pages: int = Form(1),
    min_pages_per_chunk: int = Form(1),
    respect_boundaries: bool = Form(True),
    max_workers: int = Form(4),
    
    # Embedding parameters (shared)
    embed_batch_size: int = Form(4),
    embed_delay_between_requests: float = Form(0.2),
    embed_max_retries: int = Form(3),
    embed_retry_delay: float = Form(20.0),
    
    # Sub-intent parameters (unstructured)
    sub_intent_similarity_threshold: float = Form(0.75),
    sub_intent_batch_size: int = Form(4),
    sub_intent_delay_between_requests: float = Form(0.2),
    sub_intent_max_retries: int = Form(3),
    sub_intent_retry_delay: float = Form(20.0),
    
    # Intent mapping parameters (shared)
    intent_similarity_threshold: float = Form(0.75),
    top_n_candidates: int = Form(10),
    intent_batch_size: int = Form(20),
    intent_delay: float = Form(0.2),
    
    # Semi-structured pipeline specific parameters
    preserve_layout_alignment_across_pages: bool = Form(False),
    skip_diagonal_text: bool = Form(True),
    output_tables_as_HTML: bool = Form(False),
    disable_image_extraction: bool = Form(False),
    spreadsheet_extract_sub_tables: bool = Form(True),
    result_type: str = Form("markdown"),
    chunk_size: int = Form(20000),
    chunk_overlap: int = Form(1000),
    
    # Semi-structured specific parameters (different from unstructured)
    semi_embed_batch_size: int = Form(4),
    semi_embed_delay_between_requests: float = Form(0.3),
    semi_embed_max_retries: int = Form(3),
    semi_embed_retry_delay: float = Form(1.0),
    
    # Semi-structured sub-intent parameters (different names than unstructured)
    semi_similarity_threshold: float = Form(0.75),
    semi_sub_intent_batch_size: int = Form(4),
    semi_sub_intent_delay_between_requests: float = Form(0.2),
    semi_sub_intent_max_retries: int = Form(3),
    semi_sub_intent_retry_delay: float = Form(1.0),
    
    # Semi-structured intent mapping parameters (different from unstructured)
    semi_intent_similarity_threshold: float = Form(0.75),
    semi_top_n_candidates: int = Form(10),
    semi_intent_batch_size: int = Form(8),
    semi_intent_delay: float = Form(0.2),
    
    # NEW PARAMETER: Process files individually or in batches
    process_individually: bool = Form(True, description="Process each file individually (recommended for tracking)"),
    
    # --- UPDATED FIELDS - Now accept JSON arrays ---
    file_descriptions: str = Form(None, description="JSON array of descriptions or single string: [\"desc1\", \"desc2\"] or \"single desc\""),
    table_names: str = Form(None, description="JSON array of table names or single string: [\"table1\", \"table2\"] or \"single table\""),
    user_ids: str = Form(None, description="JSON array of user_ids or single string: [\"user1\", \"user2\"] or \"user1\"")
):
    """
    Smart file processing endpoint that automatically routes files to appropriate pipelines:
    - Excel/Spreadsheet files (.xlsx, .xls, .csv) → Semi-structured pipeline
    - Other files (.pdf, .docx, .txt, etc.) → Unstructured pipeline
    
    By default, ALL files are processed individually for better tracking.
    
    file_descriptions and table_names can be:
    - JSON arrays: ["desc1", "desc2", "desc3"] - one per file
    - Single strings: "description for all files" - applied to all files
    """
    
    # Define file extensions for semi-structured processing
    SEMI_STRUCTURED_EXTENSIONS = {'.xlsx', '.xls', '.csv', '.ods'}
    
    # Separate files by processing type
    semi_structured_files = []
    unstructured_files = []
    
    for file in files:
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension in SEMI_STRUCTURED_EXTENSIONS:
                semi_structured_files.append(file)
            else:
                unstructured_files.append(file)
        else:
            # If no filename, default to unstructured
            unstructured_files.append(file)
    
    # Parse metadata for all files
    file_metadata_tuples = parse_file_metadata_arrays(file_descriptions, table_names, user_ids, len(files))
    
    results = {
        "message": "Smart file processing initiated",
        "total_files": len(files),
        "semi_structured_files": len(semi_structured_files),
        "unstructured_files": len(unstructured_files),
        "processing_mode": "individual" if process_individually else "batch",
        "task_ids": []
    }
    
    try:
        file_index = 0  # Track position in original file list
        
        # Process semi-structured files (always individual since endpoint only accepts single file)
        for file in semi_structured_files:
            # Find this file's index in the original files list
            while file_index < len(files) and files[file_index] != file:
                file_index += 1
            
            file_desc, table_name, user_id = file_metadata_tuples[file_index]
            
            response = await semi_structured_file_system_endpoint(
                background_tasks=background_tasks,
                file=file,
                preserve_filename=preserve_filenames,
                preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
                skip_diagonal_text=skip_diagonal_text,
                output_tables_as_HTML=output_tables_as_HTML,
                disable_image_extraction=disable_image_extraction,
                spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
                result_type=result_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_batch_size=semi_embed_batch_size,
                embed_delay_between_requests=semi_embed_delay_between_requests,
                embed_max_retries=semi_embed_max_retries,
                embed_retry_delay=semi_embed_retry_delay,
                similarity_threshold=semi_similarity_threshold,
                sub_intent_batch_size=semi_sub_intent_batch_size,
                sub_intent_delay_between_requests=semi_sub_intent_delay_between_requests,
                sub_intent_max_retries=semi_sub_intent_max_retries,
                sub_intent_retry_delay=semi_sub_intent_retry_delay,
                intent_similarity_threshold=semi_intent_similarity_threshold,
                top_n_candidates=semi_top_n_candidates,
                intent_batch_size=semi_intent_batch_size,
                intent_delay=semi_intent_delay,
                file_description=file_desc,
                table_name=table_name,
                user_id=user_id
            )
            
            results["task_ids"].append({
                "task_id": response.task_id,
                "pipeline": "semi_structured",
                "filename": file.filename,
                "file_description": file_desc,
                "table_name": table_name,
                "user_id": user_id,
                "status": response.status
            })
            
            file_index += 1
        
        # Reset file index for unstructured files
        file_index = 0
        
        # Process unstructured files
        if unstructured_files:
            if process_individually:
                # Process each unstructured file individually
                for file in unstructured_files:
                    # Find this file's index in the original files list
                    while file_index < len(files) and files[file_index] != file:
                        file_index += 1
                    
                    file_desc, table_name, user_id = file_metadata_tuples[file_index]
                    
                    response = await unstructured_file_system_endpoint(
                        background_tasks=background_tasks,
                        files=[file],  # Pass as single-item list
                        preserve_filenames=preserve_filenames,
                        delay_between_files=delay_between_files,
                        max_pages_per_chunk=max_pages_per_chunk,
                        boundary_sentences=boundary_sentences,
                        boundary_table_rows=boundary_table_rows,
                        target_pages_per_chunk=target_pages_per_chunk,
                        overlap_pages=overlap_pages,
                        min_pages_per_chunk=min_pages_per_chunk,
                        respect_boundaries=respect_boundaries,
                        max_workers=max_workers,
                        embed_batch_size=embed_batch_size,
                        embed_delay_between_requests=embed_delay_between_requests,
                        embed_max_retries=embed_max_retries,
                        embed_retry_delay=embed_retry_delay,
                        sub_intent_similarity_threshold=sub_intent_similarity_threshold,
                        sub_intent_batch_size=sub_intent_batch_size,
                        sub_intent_delay_between_requests=sub_intent_delay_between_requests,
                        sub_intent_max_retries=sub_intent_max_retries,
                        sub_intent_retry_delay=sub_intent_retry_delay,
                        intent_similarity_threshold=intent_similarity_threshold,
                        top_n_candidates=top_n_candidates,
                        intent_batch_size=intent_batch_size,
                        intent_delay=intent_delay,
                        file_description=file_desc,
                        table_name=table_name,
                        user_id=user_id
                    )
                    
                    results["task_ids"].append({
                        "task_id": response.task_id,
                        "pipeline": "unstructured",
                        "filename": file.filename,
                        "file_description": file_desc,
                        "table_name": table_name,
                        "user_id": user_id,
                        "status": response.status
                    })
                    
                    file_index += 1
            else:
                # For batch processing, we'll use the first file's metadata for all
                # (This is a fallback case - individual processing is recommended)
                first_desc, first_table, first_user_id = file_metadata_tuples[0] if file_metadata_tuples else (None, None, None)
                
                response = await unstructured_file_system_endpoint(
                    background_tasks=background_tasks,
                    files=unstructured_files,
                    preserve_filenames=preserve_filenames,
                    delay_between_files=delay_between_files,
                    max_pages_per_chunk=max_pages_per_chunk,
                    boundary_sentences=boundary_sentences,
                    boundary_table_rows=boundary_table_rows,
                    target_pages_per_chunk=target_pages_per_chunk,
                    overlap_pages=overlap_pages,
                    min_pages_per_chunk=min_pages_per_chunk,
                    respect_boundaries=respect_boundaries,
                    max_workers=max_workers,
                    embed_batch_size=embed_batch_size,
                    embed_delay_between_requests=embed_delay_between_requests,
                    embed_max_retries=embed_max_retries,
                    embed_retry_delay=embed_retry_delay,
                    sub_intent_similarity_threshold=sub_intent_similarity_threshold,
                    sub_intent_batch_size=sub_intent_batch_size,
                    sub_intent_delay_between_requests=sub_intent_delay_between_requests,
                    sub_intent_max_retries=sub_intent_max_retries,
                    sub_intent_retry_delay=sub_intent_retry_delay,
                    intent_similarity_threshold=intent_similarity_threshold,
                    top_n_candidates=top_n_candidates,
                    intent_batch_size=intent_batch_size,
                    intent_delay=intent_delay,
                    file_description=first_desc,
                    table_name=first_table,
                    user_id=first_user_id
                )
                
                results["task_ids"].append({
                    "task_id": response.task_id,
                    "pipeline": "unstructured",
                    "filenames": [f.filename for f in unstructured_files],
                    "file_count": len(unstructured_files),
                    "file_description": first_desc,
                    "table_name": first_table,
                    "user_id": first_user_id,
                    "status": response.status
                })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart file processing failed: {str(e)}")

@app.post("/smart_file_system", response_model=Dict, summary="Upload multiple files and automatically route to appropriate processing pipeline")
async def smart_file_system_endpoint(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple files to upload and process automatically"),
    file_descriptions: str = Form(None, description="JSON array of descriptions or single string: [\"desc1\", \"desc2\"] or \"single desc\""),
    table_names: str = Form(None, description="JSON array of table names or single string: [\"table1\", \"table2\"] or \"single table\""),
    user_ids: str = Form(None, description="JSON array of user_ids or single string: [\"user1\", \"user2\"] or \"user1\""),
):
    """
    Smart file processing endpoint that automatically routes files to appropriate pipelines:
    - Excel/Spreadsheet files (.xlsx, .xls, .csv) → Semi-structured pipeline
    - Other files (.pdf, .docx, .txt, etc.) → Unstructured pipeline
    
    All parameters are predefined for optimal processing.
    
    file_descriptions and table_names can be:
    - JSON arrays: ["desc1", "desc2", "desc3"] - one per file
    - Single strings: "description for all files" - applied to all files
    """
    
    # Predefined optimal parameters
    preserve_filenames = True
    process_individually = True
    
    # Common parameters
    delay_between_files = 1.0
    
    # Unstructured pipeline parameters
    max_pages_per_chunk = 10
    boundary_sentences = 3
    boundary_table_rows = 3
    target_pages_per_chunk = 3
    overlap_pages = 1
    min_pages_per_chunk = 1
    respect_boundaries = True
    max_workers = 4
    
    # Embedding parameters (shared)
    embed_batch_size = 4
    embed_delay_between_requests = 4.0
    embed_max_retries = 3
    embed_retry_delay = 20.0
    
    # Sub-intent parameters (unstructured)
    sub_intent_similarity_threshold = 0.75
    sub_intent_batch_size = 4
    sub_intent_delay_between_requests = 3.0
    sub_intent_max_retries = 3
    sub_intent_retry_delay = 20.0
    
    # Intent mapping parameters (shared)
    intent_similarity_threshold = 0.75
    top_n_candidates = 10
    intent_batch_size = 20
    intent_delay = 1.0
    
    # Semi-structured pipeline parameters
    preserve_layout_alignment_across_pages = False
    skip_diagonal_text = True
    output_tables_as_HTML = False
    disable_image_extraction = False
    spreadsheet_extract_sub_tables = True
    result_type = "markdown"
    chunk_size = 20000
    chunk_overlap = 1000
    
    # Semi-structured specific parameters
    semi_embed_batch_size = 4
    semi_embed_delay_between_requests = 3
    semi_embed_max_retries = 3
    semi_embed_retry_delay = 1.0
    
    # Semi-structured sub-intent parameters
    semi_similarity_threshold = 0.75
    semi_sub_intent_batch_size = 4
    semi_sub_intent_delay_between_requests = 1
    semi_sub_intent_max_retries = 3
    semi_sub_intent_retry_delay = 1.0
    
    # Semi-structured intent mapping parameters
    semi_intent_similarity_threshold = 0.75
    semi_top_n_candidates = 10
    semi_intent_batch_size = 8
    semi_intent_delay = 1
    
    # Define file extensions for semi-structured processing
    SEMI_STRUCTURED_EXTENSIONS = {'.xlsx', '.xls', '.csv', '.ods'}
    
    # Separate files by processing type
    semi_structured_files = []
    unstructured_files = []
    
    for file in files:
        if file.filename:
            file_extension = Path(file.filename).suffix.lower()
            if file_extension in SEMI_STRUCTURED_EXTENSIONS:
                semi_structured_files.append(file)
            else:
                unstructured_files.append(file)
        else:
            # If no filename, default to unstructured
            unstructured_files.append(file)
    
    # Parse metadata for all files
    file_metadata_tuples = parse_file_metadata_arrays(file_descriptions, table_names, user_ids, len(files))
    
    # CREATE BUNDLE TASK
    bundle_id = str(uuid.uuid4())
    all_filenames = [f.filename for f in files]
    
    results = {
        "message": "Smart file processing initiated",
        "bundle_id": bundle_id,
        "total_files": len(files),
        "semi_structured_files": len(semi_structured_files),
        "unstructured_files": len(unstructured_files),
        "processing_mode": "individual",
        "task_ids": []
    }
    
    try:
        task_ids = []
        file_index = 0  # Track position in original file list
        
        # Process semi-structured files (always individual)
        for file in semi_structured_files:
            # Find this file's index in the original files list
            while file_index < len(files) and files[file_index] != file:
                file_index += 1
            
            file_desc, table_name, user_id = file_metadata_tuples[file_index]
            
            response = await semi_structured_file_system_endpoint(
                background_tasks=background_tasks,
                file=file,
                preserve_filename=preserve_filenames,
                preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
                skip_diagonal_text=skip_diagonal_text,
                output_tables_as_HTML=output_tables_as_HTML,
                disable_image_extraction=disable_image_extraction,
                spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
                result_type=result_type,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embed_batch_size=semi_embed_batch_size,
                embed_delay_between_requests=semi_embed_delay_between_requests,
                embed_max_retries=semi_embed_max_retries,
                embed_retry_delay=semi_embed_retry_delay,
                similarity_threshold=semi_similarity_threshold,
                sub_intent_batch_size=semi_sub_intent_batch_size,
                sub_intent_delay_between_requests=semi_sub_intent_delay_between_requests,
                sub_intent_max_retries=semi_sub_intent_max_retries,
                sub_intent_retry_delay=semi_sub_intent_retry_delay,
                intent_similarity_threshold=semi_intent_similarity_threshold,
                top_n_candidates=semi_top_n_candidates,
                intent_batch_size=semi_intent_batch_size,
                intent_delay=semi_intent_delay,
                file_description=file_desc,
                table_name=table_name,
                user_id=user_id
            )
            
            task_ids.append(response.task_id)
            results["task_ids"].append({
                "task_id": response.task_id,
                "pipeline": "semi_structured",
                "filename": file.filename,
                "file_description": file_desc,
                "table_name": table_name,
                "user_id": user_id,
                "status": response.status
            })
            
            file_index += 1
        
        # Reset file index for unstructured files
        file_index = 0
        
        # Process unstructured files individually
        for file in unstructured_files:
            # Find this file's index in the original files list
            while file_index < len(files) and files[file_index] != file:
                file_index += 1
            
            file_desc, table_name, user_id = file_metadata_tuples[file_index]
            
            response = await unstructured_file_system_endpoint(
                background_tasks=background_tasks,
                files=[file],  # Pass as single-item list
                preserve_filenames=preserve_filenames,
                delay_between_files=delay_between_files,
                max_pages_per_chunk=max_pages_per_chunk,
                boundary_sentences=boundary_sentences,
                boundary_table_rows=boundary_table_rows,
                target_pages_per_chunk=target_pages_per_chunk,
                overlap_pages=overlap_pages,
                min_pages_per_chunk=min_pages_per_chunk,
                respect_boundaries=respect_boundaries,
                max_workers=max_workers,
                embed_batch_size=embed_batch_size,
                embed_delay_between_requests=embed_delay_between_requests,
                embed_max_retries=embed_max_retries,
                embed_retry_delay=embed_retry_delay,
                sub_intent_similarity_threshold=sub_intent_similarity_threshold,
                sub_intent_batch_size=sub_intent_batch_size,
                sub_intent_delay_between_requests=sub_intent_delay_between_requests,
                sub_intent_max_retries=sub_intent_max_retries,
                sub_intent_retry_delay=sub_intent_retry_delay,
                intent_similarity_threshold=intent_similarity_threshold,
                top_n_candidates=top_n_candidates,
                intent_batch_size=intent_batch_size,
                intent_delay=intent_delay,
                file_description=file_desc,
                table_name=table_name,
                user_id=user_id
            )
            
            task_ids.append(response.task_id)
            results["task_ids"].append({
                "task_id": response.task_id,
                "pipeline": "unstructured",
                "filename": file.filename,
                "file_description": file_desc,
                "table_name": table_name,
                "user_id": user_id,
                "status": response.status
            })
            
            file_index += 1
        
        # CREATE AND STORE BUNDLE TASK
        bundle_task = BundleTaskInfo(
            bundle_id=bundle_id,
            task_ids=task_ids,
            total_files=len(files),
            filenames=all_filenames,
            created_at=datetime.now()
        )
        
        db_manager.create_bundle(bundle_task)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart file processing failed: {str(e)}")

# Add these imports at the top of your file
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import uuid
from threading import Lock

# Add these new data structures after your existing TaskInfo class

class BundleTaskInfo:
    """Information about a bundle of tasks"""
    def __init__(self, bundle_id: str, task_ids: List[str], total_files: int, 
                 filenames: List[str], created_at: datetime):
        self.bundle_id = bundle_id
        self.task_ids = task_ids
        self.total_files = total_files
        self.filenames = filenames
        self.created_at = created_at
        self.completed_files = 0
        self.failed_files = 0
        self.current_processing_files = []
        self.remaining_files = filenames.copy()
        self.status = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
        self.progress_percentage = 0.0
        self.last_updated = created_at

class BundleTaskResponse:
    """Response model for bundle task status"""
    def __init__(self, bundle_task: BundleTaskInfo, individual_tasks: List[TaskInfo]):
        self.bundle_id = bundle_task.bundle_id
        self.status = bundle_task.status
        self.total_files = bundle_task.total_files
        self.completed_files = bundle_task.completed_files
        self.failed_files = bundle_task.failed_files
        self.remaining_files = len(bundle_task.remaining_files)
        self.progress_percentage = bundle_task.progress_percentage
        self.created_at = bundle_task.created_at
        self.last_updated = bundle_task.last_updated
        self.current_processing_files = bundle_task.current_processing_files
        self.remaining_file_names = bundle_task.remaining_files
        self.individual_tasks = individual_tasks

# Add bundle storage (add this near your existing task_store)
# bundle_store: Dict[str, BundleTaskInfo] = {}
# bundle_lock = Lock()

# Helper function to update bundle status
def update_bundle_status(bundle_id: str):
    """Update bundle status based on individual task statuses"""
    bundle_info = db_manager.get_bundle(bundle_id)
    if not bundle_info:
        return
    
    # Get status of all individual tasks
    individual_statuses = []
    current_processing = []
    completed_files = []
    failed_files = []
    
    for task_id in bundle_info.task_ids:
        task_info = db_manager.get_task(task_id)
        if task_info:
            individual_statuses.append(task_info.status)
            
            if task_info.status == TaskStatus.RUNNING:
                if task_info.current_file:
                    current_processing.append(task_info.current_file)
            elif task_info.status == TaskStatus.COMPLETED:
                if task_info.current_file:
                    completed_files.append(task_info.current_file)
            elif task_info.status == TaskStatus.FAILED:
                if task_info.current_file:
                    failed_files.append(task_info.current_file)
    
    # Update bundle status
    bundle_info.completed_files = len(completed_files)
    bundle_info.failed_files = len(failed_files)
    bundle_info.current_processing_files = current_processing
    bundle_info.last_updated = datetime.now()
    
    # Update remaining files - find pending tasks
    pending_tasks = []
    for task_id in bundle_info.task_ids:
        task_info = db_manager.get_task(task_id)
        if task_info and task_info.status == TaskStatus.PENDING:
            # Find original filename that corresponds to this task
            original_filename = task_info.current_file
            for filename in bundle_info.filenames:
                if filename in task_info.current_file or task_info.current_file in filename:
                    original_filename = filename
                    break
            pending_tasks.append(original_filename)

    bundle_info.remaining_files = pending_tasks
    
    # Calculate progress
    total_processed = bundle_info.completed_files + bundle_info.failed_files
    bundle_info.progress_percentage = (total_processed / bundle_info.total_files) * 100 if bundle_info.total_files > 0 else 0
    
    # Determine overall status
    if all(status == TaskStatus.COMPLETED for status in individual_statuses):
        bundle_info.status = "COMPLETED"
    elif any(status == TaskStatus.FAILED for status in individual_statuses) and \
         all(status in [TaskStatus.COMPLETED, TaskStatus.FAILED] for status in individual_statuses):
        bundle_info.status = "FAILED"
    elif any(status == TaskStatus.RUNNING for status in individual_statuses):
        bundle_info.status = "RUNNING"
    else:
        bundle_info.status = "PENDING"
    
    # Update bundle in database
    db_manager.update_bundle(
        bundle_id,
        status=bundle_info.status,
        completed_files=bundle_info.completed_files,
        failed_files=bundle_info.failed_files,
        current_processing_files=bundle_info.current_processing_files,
        remaining_files=bundle_info.remaining_files,
        progress_percentage=bundle_info.progress_percentage,
        last_updated=bundle_info.last_updated
    )

# NEW ENDPOINT: Get bundle task status
@app.get("/bundle_task_status/{bundle_id}", summary="Get bundle task status with all details")
async def get_bundle_task_status(bundle_id: str):
    """
    Get comprehensive status of a bundle task including:
    - Overall progress and status
    - Files currently being processed
    - Files remaining
    - Individual task details
    """
    
    # Update bundle status first
    update_bundle_status(bundle_id)
    
    # with bundle_lock: # This line is removed as per the new_code
    #     if bundle_id not in bundle_store: # This line is removed as per the new_code
    #         raise HTTPException(status_code=404, detail="Bundle task not found") # This line is removed as per the new_code
    
    # bundle = bundle_store[bundle_id] # This line is removed as per the new_code
    
    # # Get individual task details # This line is removed as per the new_code
    # individual_tasks = [] # This line is removed as per the new_code
    # with task_lock: # This line is removed as per the new_code
    #     for task_id in bundle.task_ids: # This line is removed as per the new_code
    #         if task_id in task_store: # This line is removed as per the new_code
    #             individual_tasks.append(task_store[task_id]) # This line is removed as per the new_code
        
    # # Create response # This line is removed as per the new_code
    # response = BundleTaskResponse(bundle, individual_tasks) # This line is removed as per the new_code
    
    # Use database manager
    with db_manager.get_connection() as conn:
        bundle_info = db_manager.get_bundle(bundle_id)
        if not bundle_info:
            raise HTTPException(status_code=404, detail="Bundle task not found")
        
        # Get individual task details
        individual_tasks = []
        with db_manager.get_connection() as conn_inner:
            for task_id in bundle_info.task_ids:
                task_row = conn_inner.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,)).fetchone()
                if task_row:
                    individual_tasks.append(db_manager._row_to_task_info(task_row))
        
        # Create response
        response = BundleTaskResponse(bundle_info, individual_tasks)
        
        return {
            "bundle_id": response.bundle_id,
            "status": response.status,
            "total_files": response.total_files,
            "completed_files": response.completed_files,
            "failed_files": response.failed_files,
            "remaining_files": response.remaining_files,
            "progress_percentage": round(response.progress_percentage, 2),
            "created_at": response.created_at,
            "last_updated": response.last_updated,
            "current_processing_files": response.current_processing_files,
            "remaining_file_names": response.remaining_file_names,
            "individual_tasks": [
                {
                    "task_id": task.task_id,
                    "status": task.status.value,
                    "filename": task.current_file,
                    "progress": task.progress,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "error_message": task.error
                }
                for task in response.individual_tasks
            ]
        }

# NEW ENDPOINT: Get all bundle task statuses
@app.get("/bundle_task_status_all", summary="Get status of all bundle tasks")
async def get_all_bundle_task_status():
    """
    Get status of all bundle tasks in the system.
    """
    # with bundle_lock: # This line is removed as per the new_code
    #     all_bundles = [] # This line is removed as per the new_code
    #     for bundle_id in bundle_store: # This line is removed as per the new_code
    #         # Update each bundle status # This line is removed as per the new_code
    #         update_bundle_status(bundle_id) # This line is removed as per the new_code
    #         bundle = bundle_store[bundle_id] # This line is removed as per the new_code
            
    #         all_bundles.append({ # This line is removed as per the new_code
    #             "bundle_id": bundle.bundle_id, # This line is removed as per the new_code
    #             "status": bundle.status, # This line is removed as per the new_code
    #             "total_files": bundle.total_files, # This line is removed as per the new_code
    #             "completed_files": bundle.completed_files, # This line is removed as per the new_code
    #             "failed_files": bundle.failed_files, # This line is removed as per the new_code
    #             "remaining_files": len(bundle.remaining_files), # This line is removed as per the new_code
    #             "progress_percentage": round(bundle.progress_percentage, 2), # This line is removed as per the new_code
    #             "created_at": bundle.created_at, # This line is removed as per the new_code
    #             "last_updated": bundle.last_updated, # This line is removed as per the new_code
    #             "filenames": bundle.filenames # This line is removed as per the new_code
    #         }) # This line is removed as per the new_code
        
    # Use database manager
    with db_manager.get_connection() as conn:
        all_bundles = []
        for bundle_row in conn.execute('SELECT * FROM bundles ORDER BY created_at DESC').fetchall():
            bundle_info = db_manager._row_to_bundle_info(bundle_row)
            # Update each bundle status
            update_bundle_status(bundle_info.bundle_id)
            
            all_bundles.append({
                "bundle_id": bundle_info.bundle_id,
                "status": bundle_info.status,
                "total_files": bundle_info.total_files,
                "completed_files": bundle_info.completed_files,
                "failed_files": bundle_info.failed_files,
                "remaining_files": len(bundle_info.remaining_files),
                "progress_percentage": round(bundle_info.progress_percentage, 2),
                "created_at": bundle_info.created_at,
                "last_updated": bundle_info.last_updated,
                "filenames": bundle_info.filenames
            })
        
        return {
            "total_bundles": len(all_bundles),
            "bundles": all_bundles
        }

# Database cleanup endpoints
@app.delete("/cleanup_old_tasks", summary="Clean up completed tasks")
async def cleanup_old_tasks(older_than_hours: int = 24):
    """
    Remove completed tasks older than specified hours.
    """
    cleaned_count = db_manager.cleanup_old_tasks(older_than_hours)
    return {
        "message": f"Cleaned up {cleaned_count} old tasks",
        "cleaned_count": cleaned_count
    }

@app.delete("/cleanup_old_bundles", summary="Clean up completed bundle tasks")
async def cleanup_old_bundles(older_than_hours: int = 24):
    """
    Remove completed bundle tasks older than specified hours.
    """
    cleaned_count = db_manager.cleanup_old_bundles(older_than_hours)
    return {
        "message": f"Cleaned up {cleaned_count} old bundle tasks",
        "cleaned_count": cleaned_count
    }

# Test database endpoint
@app.get("/database_status", summary="Get database status")
async def get_database_status():
    """
    Get basic statistics about the database.
    """
    with db_manager.get_connection() as conn:
        # Count tasks by status
        task_counts = {}
        for status in ["pending", "running", "completed", "failed"]:
            count = conn.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', (status,)).fetchone()[0]
            task_counts[status] = count
        
        # Count bundles by status
        bundle_counts = {}
        for status in ["PENDING", "RUNNING", "COMPLETED", "FAILED"]:
            count = conn.execute('SELECT COUNT(*) FROM bundles WHERE status = ?', (status,)).fetchone()[0]
            bundle_counts[status] = count
        
        # Get total counts
        total_tasks = conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
        total_bundles = conn.execute('SELECT COUNT(*) FROM bundles').fetchone()[0]
        
        return {
            "database_file": db_manager.db_path,
            "total_tasks": total_tasks,
            "total_bundles": total_bundles,
            "task_counts_by_status": task_counts,
            "bundle_counts_by_status": bundle_counts
        }
# Helper function to check task status across all pipelines
@app.get("/smart_task_status/{task_id}", response_model=TaskInfo, summary="Get task status for any pipeline")
async def get_smart_task_status(task_id: str):
    """
    Get task status for any pipeline (unstructured or semi-structured).
    """
    # with task_lock: # This line is removed as per the new_code
    #     if task_id not in task_store: # This line is removed as per the new_code
    #         raise HTTPException(status_code=404, detail="Task not found") # This line is removed as per the new_code
    # return task_store[task_id] # This line is removed as per the new_code

    # Use database manager
    task_info = db_manager.get_task(task_id)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_info


# Helper function to get status of all tasks from smart processing
@app.get("/smart_task_status_all", response_model=List[TaskInfo], summary="Get status of all tasks")
async def get_all_smart_task_status():
    """
    Get status of all tasks in the system.
    """
    # with task_lock: # This line is removed as per the new_code
    #     return list(task_store.values()) # This line is removed as per the new_code

    # Use database manager
    return db_manager.get_all_tasks()
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app.mount("/uploads", StaticFiles(directory="./uploads/"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=8902, log_level="info",   reload=False,workers=1)