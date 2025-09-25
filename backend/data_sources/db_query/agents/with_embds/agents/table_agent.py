"""
Vector Database for Table Schema Embeddings with Progress Tracking
- Stores table schema embeddings in SQLite with vector search capabilities
- Supports CRUD operations for table information
- Fast similarity search using cosine similarity
- Metadata storage for columns, relationships, descriptions
- Progress tracking for embedding generation and batch operations
- Optimized for table relevance detection in database queries
"""

import sqlite3
import json
import asyncio
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pickle
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")


@dataclass
class TableRecord:
    """Table record for vector database"""
    id: Optional[int] = None
    table_name: str = ""
    description: str = ""
    columns: List[Dict[str, Any]] = None
    relationships: List[Dict[str, Any]] = None
    embedding: np.ndarray = None
    combined_text: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.relationships is None:
            self.relationships = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.now().isoformat()


@dataclass
class ProgressState:
    """Progress tracking state"""
    operation_id: str
    total_items: int
    processed_items: int
    failed_items: int
    start_time: str
    last_update: str
    status: str  # 'running', 'completed', 'failed', 'paused'
    current_item: str = ""
    error_log: List[str] = None
    
    def __post_init__(self):
        if self.error_log is None:
            self.error_log = []


class ProgressTracker:
    """Handles progress tracking and persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.progress_file = db_path.replace('.db', '_progress.json')
        self._initialize_progress_table()
    
    def _initialize_progress_table(self):
        """Initialize progress tracking table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress_tracking (
                    operation_id TEXT PRIMARY KEY,
                    total_items INTEGER,
                    processed_items INTEGER,
                    failed_items INTEGER,
                    start_time TEXT,
                    last_update TEXT,
                    status TEXT,
                    current_item TEXT,
                    error_log TEXT,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to initialize progress table: {e}")
    
    def save_progress(self, progress: ProgressState):
        """Save progress to database and file"""
        try:
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO progress_tracking 
                (operation_id, total_items, processed_items, failed_items, 
                 start_time, last_update, status, current_item, error_log, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                progress.operation_id,
                progress.total_items,
                progress.processed_items,
                progress.failed_items,
                progress.start_time,
                progress.last_update,
                progress.status,
                progress.current_item,
                json.dumps(progress.error_log),
                ""  # metadata field for future use
            ))
            
            conn.commit()
            conn.close()
            
            # Also save to JSON file as backup
            with open(self.progress_file, 'w') as f:
                json.dump(asdict(progress), f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def load_progress(self, operation_id: str) -> Optional[ProgressState]:
        """Load progress from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM progress_tracking WHERE operation_id = ?', 
                (operation_id,)
            )
            row = cursor.fetchone()
            
            if row:
                progress = ProgressState(
                    operation_id=row[0],
                    total_items=row[1],
                    processed_items=row[2],
                    failed_items=row[3],
                    start_time=row[4],
                    last_update=row[5],
                    status=row[6],
                    current_item=row[7],
                    error_log=json.loads(row[8]) if row[8] else []
                )
                conn.close()
                return progress
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
            return None
    
    def get_all_operations(self) -> List[ProgressState]:
        """Get all tracked operations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM progress_tracking ORDER BY last_update DESC')
            rows = cursor.fetchall()
            
            operations = []
            for row in rows:
                operations.append(ProgressState(
                    operation_id=row[0],
                    total_items=row[1],
                    processed_items=row[2],
                    failed_items=row[3],
                    start_time=row[4],
                    last_update=row[5],
                    status=row[6],
                    current_item=row[7],
                    error_log=json.loads(row[8]) if row[8] else []
                ))
            
            conn.close()
            return operations
            
        except Exception as e:
            logger.error(f"Error getting operations: {e}")
            return []


class TableVectorDatabase:
    """Vector database for storing and searching table schema embeddings with progress tracking"""
    
    def __init__(self, db_path: str = "table_schema_vector.db"):
        self.db_path = db_path
        self.gemini_client = None
        self.progress_tracker = ProgressTracker(db_path)
        self._initialize_database()
    
    def _get_gemini_client(self):
        """Get Gemini client instance"""
        if self.gemini_client is None:
            self.gemini_client = genai.Client(api_key=gemini_apikey)
        return self.gemini_client
    
    def _initialize_database(self):
        """Initialize SQLite database with vector storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tables (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    table_name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    columns TEXT,
                    relationships TEXT,
                    embedding BLOB,
                    combined_text TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    embedding_status TEXT DEFAULT 'pending'  -- 'pending', 'completed', 'failed'
                )
            ''')
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_table_name ON tables(table_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_status ON tables(embedding_status)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Table vector database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes"""
        return pickle.dumps(embedding)
    
    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        """Deserialize bytes to numpy array"""
        return pickle.loads(blob)
    
    async def _get_embedding_async(self, text: str) -> Optional[np.ndarray]:
        """Get embedding using Gemini API"""
        try:
            client = self._get_gemini_client()
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                lambda: client.models.embed_content(
                    model="gemini-embedding-exp-03-07",
                    contents=text,
                    config=genai.types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
            )
            await asyncio.sleep(5)  # Rate limiting

            return np.array(result.embeddings[0].values)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _create_combined_text(self, table_name: str, description: str, 
                            columns: List[Dict[str, Any]], 
                            relationships: List[Dict[str, Any]]) -> str:
        """Create a semantically meaningful text passage combining table components"""
        parts = []
        parts.append(f"This is the '{table_name}' table.")
        
        if description:
            parts.append(f"It {description.lower().rstrip('.')}.")
        
        return " ".join(parts)
    
    def get_pending_embeddings(self) -> List[TableRecord]:
        """Get records that need embeddings generated"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM tables 
                WHERE embedding_status = 'pending' OR embedding IS NULL
                ORDER BY table_name
            ''')
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(TableRecord(
                    id=row[0],
                    table_name=row[1],
                    description=row[2],
                    columns=json.loads(row[3]) if row[3] else [],
                    relationships=json.loads(row[4]) if row[4] else [],
                    embedding=self._deserialize_embedding(row[5]) if row[5] else None,
                    combined_text=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                ))
            
            conn.close()
            return records
            
        except Exception as e:
            logger.error(f"Error getting pending embeddings: {e}")
            return []
    
    def update_embedding_status(self, table_name: str, status: str, embedding: Optional[np.ndarray] = None):
        """Update embedding status for a record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if embedding is not None:
                cursor.execute('''
                    UPDATE tables 
                    SET embedding_status = ?, embedding = ?, updated_at = ?
                    WHERE table_name = ?
                ''', (status, self._serialize_embedding(embedding), datetime.now().isoformat(), table_name))
            else:
                cursor.execute('''
                    UPDATE tables 
                    SET embedding_status = ?, updated_at = ?
                    WHERE table_name = ?
                ''', (status, datetime.now().isoformat(), table_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating embedding status for {table_name}: {e}")
    
    async def generate_missing_embeddings(self, operation_id: str = None, batch_size: int = 10) -> int:
        """Generate embeddings for records that don't have them, with progress tracking"""
        if operation_id is None:
            operation_id = f"table_embedding_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if we're resuming an existing operation
        existing_progress = self.progress_tracker.load_progress(operation_id)
        
        # Get pending records
        pending_records = self.get_pending_embeddings()
        
        if not pending_records:
            logger.info("No pending table embeddings found")
            return 0
        
        # Initialize or resume progress
        if existing_progress and existing_progress.status == 'running':
            logger.info(f"Resuming operation {operation_id} from {existing_progress.processed_items}/{existing_progress.total_items}")
            progress = existing_progress
            # Filter out already processed items
            processed_names = set()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT table_name FROM tables WHERE embedding_status = 'completed'")
            processed_names = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            pending_records = [r for r in pending_records if r.table_name not in processed_names]
        else:
            progress = ProgressState(
                operation_id=operation_id,
                total_items=len(pending_records),
                processed_items=0,
                failed_items=0,
                start_time=datetime.now().isoformat(),
                last_update=datetime.now().isoformat(),
                status='running'
            )
        
        logger.info(f"Starting table embedding generation for {len(pending_records)} records (Operation: {operation_id})")
        
        success_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(pending_records), batch_size):
                batch = pending_records[i:i + batch_size]
                
                for record in batch:
                    try:
                        progress.current_item = record.table_name
                        progress.last_update = datetime.now().isoformat()
                        
                        # Generate combined text if needed
                        if not record.combined_text:
                            record.combined_text = self._create_combined_text(
                                record.table_name, record.description, 
                                record.columns, record.relationships
                            )
                        
                        # Generate embedding
                        embedding = await self._get_embedding_async(record.combined_text)
                        
                        if embedding is not None:
                            # Update record with embedding
                            self.update_embedding_status(record.table_name, 'completed', embedding)
                            success_count += 1
                            progress.processed_items += 1
                            
                            logger.info(f"✅ Generated embedding for table: {record.table_name} ({progress.processed_items}/{progress.total_items})")
                        else:
                            self.update_embedding_status(record.table_name, 'failed')
                            progress.failed_items += 1
                            progress.error_log.append(f"Failed to generate embedding for {record.table_name}")
                            logger.error(f"❌ Failed to generate embedding for table: {record.table_name}")
                        
                        # Save progress every record
                        self.progress_tracker.save_progress(progress)
                        
                        # Small delay to avoid API rate limits
                        await asyncio.sleep(10)
                        
                    except Exception as e:
                        progress.failed_items += 1
                        error_msg = f"Error processing {record.table_name}: {str(e)}"
                        progress.error_log.append(error_msg)
                        logger.error(error_msg)
                        self.update_embedding_status(record.table_name, 'failed')
                
                # Save progress after each batch
                progress.last_update = datetime.now().isoformat()
                self.progress_tracker.save_progress(progress)
                
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(pending_records) + batch_size - 1)//batch_size}")
                
                # Longer delay between batches
                if i + batch_size < len(pending_records):
                    await asyncio.sleep(5.0)
            
            # Mark operation as completed
            progress.status = 'completed'
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            
            logger.info(f"✅ Table embedding generation completed! Processed: {progress.processed_items}, Failed: {progress.failed_items}")
            
        except Exception as e:
            progress.status = 'failed'
            progress.error_log.append(f"Operation failed: {str(e)}")
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            logger.error(f"Table embedding generation failed: {e}")
        
        return success_count
    
    async def batch_insert_tables_with_progress(self, records: List[TableRecord], operation_id: str = None) -> int:
        """Batch insert with progress tracking"""
        if operation_id is None:
            operation_id = f"table_batch_insert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting batch insert of {len(records)} tables (Operation: {operation_id})")
        
        # Initialize progress
        progress = ProgressState(
            operation_id=operation_id,
            total_items=len(records),
            processed_items=0,
            failed_items=0,
            start_time=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
            status='running'
        )
        
        success_count = 0
        
        try:
            # First, insert records without embeddings
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for record in records:
                try:
                    progress.current_item = record.table_name
                    
                    # Generate combined text
                    if not record.combined_text:
                        record.combined_text = self._create_combined_text(
                            record.table_name, record.description,
                            record.columns, record.relationships
                        )
                    
                    record.updated_at = datetime.now().isoformat()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO tables 
                        (table_name, description, columns, relationships, embedding, combined_text, 
                         created_at, updated_at, embedding_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.table_name,
                        record.description,
                        json.dumps(record.columns),
                        json.dumps(record.relationships),
                        self._serialize_embedding(record.embedding) if record.embedding is not None else None,
                        record.combined_text,
                        record.created_at,
                        record.updated_at,
                        'completed' if record.embedding is not None else 'pending'
                    ))
                    
                    success_count += 1
                    progress.processed_items += 1
                    progress.last_update = datetime.now().isoformat()
                    
                    # Save progress periodically
                    if progress.processed_items % 10 == 0:
                        self.progress_tracker.save_progress(progress)
                        logger.info(f"Inserted {progress.processed_items}/{progress.total_items} table records")
                
                except Exception as e:
                    progress.failed_items += 1
                    progress.error_log.append(f"Failed to insert {record.table_name}: {str(e)}")
                    logger.error(f"Error inserting {record.table_name}: {e}")
            
            conn.commit()
            conn.close()
            
            # Mark operation as completed
            progress.status = 'completed'
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            
            logger.info(f"✅ Table batch insert completed! Inserted: {success_count}, Failed: {progress.failed_items}")
            
        except Exception as e:
            progress.status = 'failed'
            progress.error_log.append(f"Batch insert failed: {str(e)}")
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            logger.error(f"Table batch insert failed: {e}")
        
        return success_count
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of all operations"""
        operations = self.progress_tracker.get_all_operations()
        
        summary = {
            "total_operations": len(operations),
            "running_operations": len([op for op in operations if op.status == 'running']),
            "completed_operations": len([op for op in operations if op.status == 'completed']),
            "failed_operations": len([op for op in operations if op.status == 'failed']),
            "operations": []
        }
        
        for op in operations[:10]:  # Show last 10 operations
            summary["operations"].append({
                "operation_id": op.operation_id,
                "status": op.status,
                "progress": f"{op.processed_items}/{op.total_items}",
                "success_rate": f"{((op.processed_items / op.total_items) * 100):.1f}%" if op.total_items > 0 else "0%",
                "last_update": op.last_update,
                "failed_items": op.failed_items
            })
        
        return summary
    
    # Include other essential methods
    async def insert_table(self, record: TableRecord) -> bool:
        """Insert a table record into the database"""
        try:
            if not record.combined_text:
                record.combined_text = self._create_combined_text(
                    record.table_name, record.description, 
                    record.columns, record.relationships
                )
            
            if record.embedding is None:
                record.embedding = await self._get_embedding_async(record.combined_text)
                if record.embedding is None:
                    logger.error(f"Failed to generate embedding for {record.table_name}")
                    return False
            
            record.updated_at = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO tables 
                (table_name, description, columns, relationships, embedding, combined_text, 
                 created_at, updated_at, embedding_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.table_name,
                record.description,
                json.dumps(record.columns),
                json.dumps(record.relationships),
                self._serialize_embedding(record.embedding),
                record.combined_text,
                record.created_at,
                record.updated_at,
                'completed'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Inserted table: {record.table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting table {record.table_name}: {e}")
            return False
    
    def get_all_tables(self) -> List[TableRecord]:
        """Retrieve all tables from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tables ORDER BY table_name')
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(TableRecord(
                    id=row[0],
                    table_name=row[1],
                    description=row[2],
                    columns=json.loads(row[3]) if row[3] else [],
                    relationships=json.loads(row[4]) if row[4] else [],
                    embedding=self._deserialize_embedding(row[5]) if row[5] else None,
                    combined_text=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                ))
            
            conn.close()
            logger.info(f"Retrieved {len(records)} tables from database")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving tables: {e}")
            return []
    
    def get_table_by_name(self, table_name: str) -> Optional[TableRecord]:
        """Get specific table by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM tables WHERE table_name = ?', (table_name,))
            row = cursor.fetchone()
            
            if row:
                record = TableRecord(
                    id=row[0],
                    table_name=row[1],
                    description=row[2],
                    columns=json.loads(row[3]) if row[3] else [],
                    relationships=json.loads(row[4]) if row[4] else [],
                    embedding=self._deserialize_embedding(row[5]) if row[5] else None,
                    combined_text=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                )
                conn.close()
                return record
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving table {table_name}: {e}")
            return None
    
    async def search_relevant_tables(self, query: str, confidence_threshold: float = 0.7, limit: int = 10) -> List[Tuple[TableRecord, float]]:
        """Search for relevant tables using vector similarity"""
        try:
            query_embedding = await self._get_embedding_async(query)
            if query_embedding is None:
                logger.error("Failed to get query embedding")
                return []
            
            all_records = self.get_all_tables()
            if not all_records:
                logger.warning("No tables found in database")
                return []
            
            results = []
            query_embedding = query_embedding.reshape(1, -1)
            
            for record in all_records:
                if record.embedding is not None:
                    record_embedding = record.embedding.reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, record_embedding)[0][0]
                    
                    if similarity >= confidence_threshold:
                        results.append((record, float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
            
            logger.info(f"Found {len(results)} relevant tables above {confidence_threshold} threshold")
            return results
            
        except Exception as e:
            logger.error(f"Error searching relevant tables: {e}")
            return []
    
    def delete_table(self, table_name: str) -> bool:
        """Delete a table by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM tables WHERE table_name = ?', (table_name,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Deleted table: {table_name}")
                return True
            else:
                logger.warning(f"Table not found: {table_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting table {table_name}: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all tables from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM tables')
            cursor.execute('DELETE FROM progress_tracking')
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted_count} tables from database")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
 
    # Move this method INSIDE the TableVectorDatabase class, 
    # right after the clear_database method (around line 550)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count total records
            cursor.execute('SELECT COUNT(*) FROM tables')
            total_count = cursor.fetchone()[0]
            
            # Count by embedding status
            cursor.execute('SELECT embedding_status, COUNT(*) FROM tables GROUP BY embedding_status')
            status_counts = dict(cursor.fetchall())
            
            # Get latest update time
            cursor.execute('SELECT MAX(updated_at) FROM tables')
            latest_update = cursor.fetchone()[0]
            
            # Get unique relationship count
            cursor.execute('SELECT DISTINCT relationships FROM tables WHERE relationships IS NOT NULL')
            rel_data = cursor.fetchall()
            unique_relationships = set()
            for row in rel_data:
                if row[0]:
                    try:
                        relationships = json.loads(row[0])
                        for rel in relationships:
                            if rel.get('related_table'):
                                unique_relationships.add(rel['related_table'])
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            # Get column statistics
            cursor.execute('SELECT columns FROM tables WHERE columns IS NOT NULL')
            col_data = cursor.fetchall()
            total_columns = 0
            unique_column_types = set()
            for row in col_data:
                if row[0]:
                    try:
                        columns = json.loads(row[0])
                        total_columns += len(columns)
                        for col in columns:
                            if col.get('data_type'):
                                unique_column_types.add(col['data_type'])
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            # Get embedding statistics
            cursor.execute('SELECT COUNT(*) FROM tables WHERE embedding IS NOT NULL')
            embedded_count = cursor.fetchone()[0]
            
            conn.close()
            
            stats = {
                "total_tables": total_count,
                "embedding_status": status_counts,
                "embedded_tables": embedded_count,
                "embedding_completion_rate": f"{(embedded_count/total_count*100):.1f}%" if total_count > 0 else "0%",
                "unique_related_tables": len(unique_relationships),
                "total_columns": total_columns,
                "unique_column_types": len(unique_column_types),
                "column_types": list(unique_column_types),
                "latest_update": latest_update,
                "database_file": self.db_path,
                "progress_summary": self.get_progress_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}

# UTILITY FUNCTIONS FOR TABLE AGENT

async def load_schema_json_to_vector_db_with_progress(schema_json_path: str, db_path: str = "table_schema_vector.db", operation_id: str = None) -> int:
    """Load table schemas from JSON file into vector database with progress tracking"""
    try:
        with open(schema_json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        tables_data = json_data.get("tables", [])
        
        # Convert to TableRecord objects
        records = []
        for table in tables_data:
            if table.get('table_name'):
                record = TableRecord(
                    table_name=table.get('table_name', ''),
                    description=table.get('description', ''),
                    columns=table.get('columns', []),
                    relationships=table.get('relationships', [])
                )
                records.append(record)
        
        # Insert into database with progress tracking
        db = TableVectorDatabase(db_path)
        success_count = await db.batch_insert_tables_with_progress(records, operation_id)
        
        logger.info(f"Loaded {success_count} tables from {schema_json_path} into {db_path}")
        return success_count
        
    except Exception as e:
        logger.error(f"Error loading schema JSON to vector DB: {e}")
        return 0


def sync_load_schema_json_to_vector_db_with_progress(schema_json_path: str, db_path: str = "table_schema_vector.db", operation_id: str = None) -> int:
    """Synchronous wrapper for loading schema JSON to vector database with progress tracking"""
    return asyncio.run(load_schema_json_to_vector_db_with_progress(schema_json_path, db_path, operation_id))


async def resume_table_embedding_generation(db_path: str = "table_schema_vector.db", operation_id: str = None) -> int:
    """Resume or start embedding generation for table records without embeddings"""
    db = TableVectorDatabase(db_path)
    return await db.generate_missing_embeddings(operation_id)


def sync_resume_table_embedding_generation(db_path: str = "table_schema_vector.db", operation_id: str = None) -> int:
    """Synchronous wrapper for resuming table embedding generation"""
    return asyncio.run(resume_table_embedding_generation(db_path, operation_id))


async def search_tables_by_query_async(query: str, db_path: str = "table_schema_vector.db", 
                                     confidence_threshold: float = 0.7, limit: int = 10) -> List[str]:
    """Search for relevant tables using vector similarity - async version"""
    try:
        db = TableVectorDatabase(db_path)
        results = await db.search_relevant_tables(query, confidence_threshold, limit)
        
        # Extract just the table names
        table_names = [record.table_name for record, similarity in results]
        
        logger.info(f"Found {len(table_names)} relevant tables for query: '{query}'")
        return table_names
        
    except Exception as e:
        logger.error(f"Error searching tables by query: {e}")
        return []


def search_tables_by_query_sync(query: str, db_path: str = "table_schema_vector.db", 
                               confidence_threshold: float = 0.7, limit: int = 10) -> List[str]:
    """Search for relevant tables using vector similarity - sync version"""
    return asyncio.run(search_tables_by_query_async(query, db_path, confidence_threshold, limit))


def get_table_database_info(db_path: str = "table_schema_vector.db") -> Dict[str, Any]:
    """Get comprehensive information about the table vector database"""
    try:
        db = TableVectorDatabase(db_path)
        stats = db.get_database_stats()
        
        # Add file size information
        if os.path.exists(db_path):
            file_size = os.path.getsize(db_path)
            stats["database_size_mb"] = round(file_size / (1024 * 1024), 2)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting table database info: {e}")
        return {}


async def enhanced_table_agent_with_vector_search(user_query: str, subintent_output: Dict[str, List[str]], 
                                                schema_json_path: str, vector_db_path: str = "table_schema_vector.db",
                                                use_vector_search: bool = True, confidence_threshold: float = 0.6) -> List[str]:
    """
    Enhanced table agent that combines original logic with vector search for better accuracy
    
    Args:
        user_query: User's database query
        subintent_output: Output from sub-intent classification
        schema_json_path: Path to schema JSON file
        vector_db_path: Path to vector database
        use_vector_search: Whether to use vector search as additional filter
        confidence_threshold: Minimum similarity score for vector search
        
    Returns:
        List of relevant table names
    """
    try:
        # First, get results from original table agent
        from without_embds.agents.subIntent_agent import extract_relevant_tables  # Import your original function
        
        original_tables = extract_relevant_tables(user_query, subintent_output, schema_json_path)
        
        if not use_vector_search:
            return original_tables
        
        # Get additional tables from vector search
        vector_tables = await search_tables_by_query_async(
            user_query, vector_db_path, confidence_threshold, limit=15
        )
        
        # Combine results (original tables have priority)
        combined_tables = list(original_tables)  # Start with original results
        
        # Add vector search results that aren't already included
        for table in vector_tables:
            if table not in combined_tables:
                combined_tables.append(table)
        
        logger.info(f"Enhanced table agent results:")
        logger.info(f"  Original agent: {len(original_tables)} tables")
        logger.info(f"  Vector search: {len(vector_tables)} tables")
        logger.info(f"  Combined: {len(combined_tables)} tables")
        
        return combined_tables
        
    except Exception as e:
        logger.error(f"Error in enhanced table agent: {e}")
        # Fallback to original method
        try:
            from your_table_agent_module import extract_relevant_tables
            return extract_relevant_tables(user_query, subintent_output, schema_json_path)
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return []


def sync_enhanced_table_agent_with_vector_search(user_query: str, subintent_output: Dict[str, List[str]], 
                                                schema_json_path: str, vector_db_path: str = "table_schema_vector.db",
                                                use_vector_search: bool = True, confidence_threshold: float = 0.6) -> List[str]:
    """Synchronous wrapper for enhanced table agent"""
    return asyncio.run(enhanced_table_agent_with_vector_search(
        user_query, subintent_output, schema_json_path, vector_db_path, use_vector_search, confidence_threshold
    ))


# BATCH OPERATIONS

async def batch_update_table_descriptions(db_path: str, table_updates: Dict[str, str], operation_id: str = None) -> int:
    """Batch update table descriptions and regenerate embeddings"""
    try:
        db = TableVectorDatabase(db_path)
        
        if operation_id is None:
            operation_id = f"table_desc_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        success_count = 0
        
        for table_name, new_description in table_updates.items():
            # Get existing record
            record = db.get_table_by_name(table_name)
            if record:
                # Update description
                record.description = new_description
                record.updated_at = datetime.now().isoformat()
                
                # Regenerate combined text and embedding
                record.combined_text = db._create_combined_text(
                    record.table_name, record.description, 
                    record.columns, record.relationships
                )
                record.embedding = await db._get_embedding_async(record.combined_text)
                
                # Update in database
                if await db.insert_table(record):
                    success_count += 1
                    logger.info(f"Updated description for table: {table_name}")
                else:
                    logger.error(f"Failed to update table: {table_name}")
            else:
                logger.warning(f"Table not found: {table_name}")
        
        logger.info(f"Updated descriptions for {success_count}/{len(table_updates)} tables")
        return success_count
        
    except Exception as e:
        logger.error(f"Error in batch update descriptions: {e}")
        return 0


def cleanup_failed_operations(db_path: str, days_old: int = 7) -> int:
    """Clean up old failed operations from progress tracking"""
    try:
        db = TableVectorDatabase(db_path)
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM progress_tracking 
            WHERE status = 'failed' AND last_update < ?
        ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_count} failed operations older than {days_old} days")
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error cleaning up failed operations: {e}")
        return 0


# EXAMPLE USAGE AND TESTING

if __name__ == "__main__":
    # Configuration
    SCHEMA_JSON_PATH = r"/Users/nilab/Desktop/projects/Knowladge-Base/all_tables_with_descriptions_v1.json"
    VECTOR_DB_PATH = "table_schema_vector.db"
    
    async def main():
        print("🚀 Setting up Table Vector Database...")
        
        # # Load schema into vector database
        # print("📥 Loading schema into vector database...")
        # loaded_count = await load_schema_json_to_vector_db_with_progress(
        #     SCHEMA_JSON_PATH, VECTOR_DB_PATH
        # )
        # print(f"✅ Loaded {loaded_count} tables")
        
        # Generate embeddings
        print("🧠 Generating embeddings...")
        embedded_count = await resume_table_embedding_generation(VECTOR_DB_PATH)
        print(f"✅ Generated embeddings for {embedded_count} tables")
        
        # Test search
        test_query = "Show me pending expenses for project dashboard"
        print(f"\n🔍 Testing search for: '{test_query}'")
        
        relevant_tables = await search_tables_by_query_async(
            test_query, VECTOR_DB_PATH, confidence_threshold=0.6
        )
        print(f"📋 Found relevant tables: {relevant_tables}")
        
        # Get database info
        print("\n📊 Database Statistics:")
        stats = get_table_database_info(VECTOR_DB_PATH)
        for key, value in stats.items():
            if key != 'progress_summary':
                print(f"  {key}: {value}")
    
    # Run the async main function
    asyncio.run(main())