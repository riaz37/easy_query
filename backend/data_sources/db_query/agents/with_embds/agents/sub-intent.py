"""
Vector Database for Sub-Intent Embeddings with Progress Tracking
- Stores embeddings in SQLite with vector search capabilities
- Supports CRUD operations for sub-intents
- Fast similarity search using cosine similarity
- Metadata storage for tables, keywords, descriptions
- Progress tracking for embedding generation and batch operations
"""

import sqlite3
import json
import asyncio
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
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
class SubIntentRecord:
    """Sub-intent record for vector database"""
    id: Optional[int] = None
    name: str = ""
    description: str = ""
    keywords: List[str] = None
    tables: List[str] = None
    embedding: np.ndarray = None
    combined_text: str = ""
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.tables is None:
            self.tables = []
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


class VectorDatabase:
    """Vector database for storing and searching sub-intent embeddings with progress tracking"""
    
    def __init__(self, db_path: str = "sub_intents_vector.db"):
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
            
            # Create sub_intents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sub_intents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    keywords TEXT,
                    tables TEXT,
                    embedding BLOB,
                    combined_text TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    embedding_status TEXT DEFAULT 'pending'  -- 'pending', 'completed', 'failed'
                )
            ''')
            
            # Create index for faster lookups
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON sub_intents(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_embedding_status ON sub_intents(embedding_status)')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Vector database initialized at {self.db_path}")
            
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
            time.sleep(5)  # sleep 1.1 seconds for example

            return np.array(result.embeddings[0].values)
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _create_combined_text(self, name: str, description: str, keywords: List[str], tables: List[str]) -> str:
        """Create a semantically meaningful text passage combining intent components"""
        parts = []
        parts.append(f"This intent is called '{name}'.")
        
        if description:
            parts.append(f"It is designed to {description.lower().rstrip('.')}.")
        
        if keywords:
            keyword_text = ", ".join(keywords)
            parts.append(f"It performs tasks such as {keyword_text}.")
        
        if tables:
            table_text = ", ".join(tables)
            parts.append(f"This intent interacts with the following database tables: {table_text}.")
        
        return " ".join(parts)
    
    def get_pending_embeddings(self) -> List[SubIntentRecord]:
        """Get records that need embeddings generated"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM sub_intents 
                WHERE embedding_status = 'pending' OR embedding IS NULL
                ORDER BY name
            ''')
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(SubIntentRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    keywords=json.loads(row[3]) if row[3] else [],
                    tables=json.loads(row[4]) if row[4] else [],
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
    
    def update_embedding_status(self, name: str, status: str, embedding: Optional[np.ndarray] = None):
        """Update embedding status for a record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if embedding is not None:
                cursor.execute('''
                    UPDATE sub_intents 
                    SET embedding_status = ?, embedding = ?, updated_at = ?
                    WHERE name = ?
                ''', (status, self._serialize_embedding(embedding), datetime.now().isoformat(), name))
            else:
                cursor.execute('''
                    UPDATE sub_intents 
                    SET embedding_status = ?, updated_at = ?
                    WHERE name = ?
                ''', (status, datetime.now().isoformat(), name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating embedding status for {name}: {e}")
    
    async def generate_missing_embeddings(self, operation_id: str = None, batch_size: int = 10) -> int:
        """Generate embeddings for records that don't have them, with progress tracking"""
        if operation_id is None:
            operation_id = f"embedding_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Check if we're resuming an existing operation
        existing_progress = self.progress_tracker.load_progress(operation_id)
        
        # Get pending records
        pending_records = self.get_pending_embeddings()
        
        if not pending_records:
            logger.info("No pending embeddings found")
            return 0
        
        # Initialize or resume progress
        if existing_progress and existing_progress.status == 'running':
            logger.info(f"Resuming operation {operation_id} from {existing_progress.processed_items}/{existing_progress.total_items}")
            progress = existing_progress
            # Filter out already processed items
            processed_names = set()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sub_intents WHERE embedding_status = 'completed'")
            processed_names = {row[0] for row in cursor.fetchall()}
            conn.close()
            
            pending_records = [r for r in pending_records if r.name not in processed_names]
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
        
        logger.info(f"Starting embedding generation for {len(pending_records)} records (Operation: {operation_id})")
        
        success_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(pending_records), batch_size):
                batch = pending_records[i:i + batch_size]
                
                for record in batch:
                    try:
                        progress.current_item = record.name
                        progress.last_update = datetime.now().isoformat()
                        
                        # Generate combined text if needed
                        if not record.combined_text:
                            record.combined_text = self._create_combined_text(
                                record.name, record.description, record.keywords, record.tables
                            )
                        
                        # Generate embedding
                        embedding = await self._get_embedding_async(record.combined_text)
                        
                        if embedding is not None:
                            # Update record with embedding
                            self.update_embedding_status(record.name, 'completed', embedding)
                            success_count += 1
                            progress.processed_items += 1
                            
                            logger.info(f"✅ Generated embedding for: {record.name} ({progress.processed_items}/{progress.total_items})")
                        else:
                            self.update_embedding_status(record.name, 'failed')
                            progress.failed_items += 1
                            progress.error_log.append(f"Failed to generate embedding for {record.name}")
                            logger.error(f"❌ Failed to generate embedding for: {record.name}")
                        
                        # Save progress every record
                        self.progress_tracker.save_progress(progress)
                        
                        # Small delay to avoid API rate limits
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        progress.failed_items += 1
                        error_msg = f"Error processing {record.name}: {str(e)}"
                        progress.error_log.append(error_msg)
                        logger.error(error_msg)
                        self.update_embedding_status(record.name, 'failed')
                
                # Save progress after each batch
                progress.last_update = datetime.now().isoformat()
                self.progress_tracker.save_progress(progress)
                
                logger.info(f"Completed batch {i//batch_size + 1}/{(len(pending_records) + batch_size - 1)//batch_size}")
                
                # Longer delay between batches
                if i + batch_size < len(pending_records):
                    await asyncio.sleep(1.0)
            
            # Mark operation as completed
            progress.status = 'completed'
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            
            logger.info(f"✅ Embedding generation completed! Processed: {progress.processed_items}, Failed: {progress.failed_items}")
            
        except Exception as e:
            progress.status = 'failed'
            progress.error_log.append(f"Operation failed: {str(e)}")
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            logger.error(f"Embedding generation failed: {e}")
        
        return success_count
    
    async def batch_insert_sub_intents_with_progress(self, records: List[SubIntentRecord], operation_id: str = None) -> int:
        """Batch insert with progress tracking"""
        if operation_id is None:
            operation_id = f"batch_insert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting batch insert of {len(records)} sub-intents (Operation: {operation_id})")
        
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
                    progress.current_item = record.name
                    
                    # Generate combined text
                    if not record.combined_text:
                        record.combined_text = self._create_combined_text(
                            record.name, record.description, record.keywords, record.tables
                        )
                    
                    record.updated_at = datetime.now().isoformat()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO sub_intents 
                        (name, description, keywords, tables, embedding, combined_text, 
                         created_at, updated_at, embedding_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.name,
                        record.description,
                        json.dumps(record.keywords),
                        json.dumps(record.tables),
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
                        logger.info(f"Inserted {progress.processed_items}/{progress.total_items} records")
                
                except Exception as e:
                    progress.failed_items += 1
                    progress.error_log.append(f"Failed to insert {record.name}: {str(e)}")
                    logger.error(f"Error inserting {record.name}: {e}")
            
            conn.commit()
            conn.close()
            
            # Mark operation as completed
            progress.status = 'completed'
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            
            logger.info(f"✅ Batch insert completed! Inserted: {success_count}, Failed: {progress.failed_items}")
            
        except Exception as e:
            progress.status = 'failed'
            progress.error_log.append(f"Batch insert failed: {str(e)}")
            progress.last_update = datetime.now().isoformat()
            self.progress_tracker.save_progress(progress)
            logger.error(f"Batch insert failed: {e}")
        
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
    
    # Include all other existing methods from the original class
    async def insert_sub_intent(self, record: SubIntentRecord) -> bool:
        """Insert a sub-intent record into the database"""
        try:
            if not record.combined_text:
                record.combined_text = self._create_combined_text(
                    record.name, record.description, record.keywords, record.tables
                )
            
            if record.embedding is None:
                record.embedding = await self._get_embedding_async(record.combined_text)
                if record.embedding is None:
                    logger.error(f"Failed to generate embedding for {record.name}")
                    return False
            
            record.updated_at = datetime.now().isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sub_intents 
                (name, description, keywords, tables, embedding, combined_text, 
                 created_at, updated_at, embedding_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.name,
                record.description,
                json.dumps(record.keywords),
                json.dumps(record.tables),
                self._serialize_embedding(record.embedding),
                record.combined_text,
                record.created_at,
                record.updated_at,
                'completed'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Inserted sub-intent: {record.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting sub-intent {record.name}: {e}")
            return False
    
    def get_all_sub_intents(self) -> List[SubIntentRecord]:
        """Retrieve all sub-intents from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM sub_intents ORDER BY name')
            rows = cursor.fetchall()
            
            records = []
            for row in rows:
                records.append(SubIntentRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    keywords=json.loads(row[3]) if row[3] else [],
                    tables=json.loads(row[4]) if row[4] else [],
                    embedding=self._deserialize_embedding(row[5]) if row[5] else None,
                    combined_text=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                ))
            
            conn.close()
            logger.info(f"Retrieved {len(records)} sub-intents from database")
            return records
            
        except Exception as e:
            logger.error(f"Error retrieving sub-intents: {e}")
            return []
    
    def get_sub_intent_by_name(self, name: str) -> Optional[SubIntentRecord]:
        """Get specific sub-intent by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM sub_intents WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            if row:
                record = SubIntentRecord(
                    id=row[0],
                    name=row[1],
                    description=row[2],
                    keywords=json.loads(row[3]) if row[3] else [],
                    tables=json.loads(row[4]) if row[4] else [],
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
            logger.error(f"Error retrieving sub-intent {name}: {e}")
            return None
    
    async def search_similar_sub_intents(self, query: str, confidence_threshold: float = 0.8, limit: int = 10) -> List[Tuple[SubIntentRecord, float]]:
        """Search for similar sub-intents using vector similarity"""
        try:
            query_embedding = await self._get_embedding_async(query)
            if query_embedding is None:
                logger.error("Failed to get query embedding")
                return []
            
            all_records = self.get_all_sub_intents()
            if not all_records:
                logger.warning("No sub-intents found in database")
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
            
            logger.info(f"Found {len(results)} similar sub-intents above {confidence_threshold} threshold")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar sub-intents: {e}")
            return []
    
    def delete_sub_intent(self, name: str) -> bool:
        """Delete a sub-intent by name"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sub_intents WHERE name = ?', (name,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Deleted sub-intent: {name}")
                return True
            else:
                logger.warning(f"Sub-intent not found: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting sub-intent {name}: {e}")
            return False
    
    def clear_database(self) -> bool:
        """Clear all sub-intents from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM sub_intents')
            cursor.execute('DELETE FROM progress_tracking')
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleared {deleted_count} sub-intents from database")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count total records
            cursor.execute('SELECT COUNT(*) FROM sub_intents')
            total_count = cursor.fetchone()[0]
            
            # Count by embedding status
            cursor.execute('SELECT embedding_status, COUNT(*) FROM sub_intents GROUP BY embedding_status')
            status_counts = dict(cursor.fetchall())
            
            # Get latest update time
            cursor.execute('SELECT MAX(updated_at) FROM sub_intents')
            latest_update = cursor.fetchone()[0]
            
            # Get unique table count
            cursor.execute('SELECT DISTINCT tables FROM sub_intents')
            table_data = cursor.fetchall()
            unique_tables = set()
            for row in table_data:
                if row[0]:
                    tables = json.loads(row[0])
                    unique_tables.update(tables)
            
            conn.close()
            
            stats = {
                "total_sub_intents": total_count,
                "embedding_status": status_counts,
                "unique_tables": len(unique_tables),
                "latest_update": latest_update,
                "database_file": self.db_path,
                "progress_summary": self.get_progress_summary()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}


# UTILITY FUNCTIONS

async def load_json_to_vector_db_with_progress(json_file_path: str, db_path: str = "sub_intents_vector.db", operation_id: str = None) -> int:
    """Load sub-intents from JSON file into vector database with progress tracking"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        sub_intents = json_data.get("sub_intents", [])
        
        # Convert to SubIntentRecord objects
        records = []
        for intent in sub_intents:
            if intent.get('name'):
                record = SubIntentRecord(
                    name=intent.get('name', ''),
                    description=intent.get('description', ''),
                    keywords=intent.get('keywords', []),
                    tables=intent.get('tables', [])
                )
                records.append(record)
        
        # Insert into database with progress tracking
        db = VectorDatabase(db_path)
        success_count = await db.batch_insert_sub_intents_with_progress(records, operation_id)
        
        logger.info(f"Loaded {success_count} sub-intents from {json_file_path} into {db_path}")
        return success_count
        
    except Exception as e:
        logger.error(f"Error loading JSON to vector DB: {e}")
        return 0

def sync_load_json_to_vector_db_with_progress(json_file_path: str, db_path: str = "sub_intents_vector.db", operation_id: str = None) -> int:
    """Synchronous wrapper for loading JSON to vector database with progress tracking"""
    return asyncio.run(load_json_to_vector_db_with_progress(json_file_path, db_path, operation_id))

async def resume_embedding_generation(db_path: str = "sub_intents_vector.db", operation_id: str = None) -> int:
    """Resume or start embedding generation for records without embeddings"""
    db = VectorDatabase(db_path)
    return await db.generate_missing_embeddings(operation_id)

def sync_resume_embedding_generation(db_path: str = "sub_intents_vector.db", operation_id: str = None) -> int:
    """Synchronous wrapper for resuming embedding generation"""
    return asyncio.run(resume_embedding_generation(db_path, operation_id))

# EXAMPLE USAGE AND TESTING
# EXAMPLE USAGE AND TESTING
if __name__ == "__main__":
    # Configuration
    JSON_FILE_PATH = r"/Users/nilab/Desktop/projects/Knowladge-Base/sub_intent_classification.json"
    DB_PATH = "sub_intents_vector.db"
    
    async def main_with_progress_and_search():
        print("🚀 Starting Vector Database Operations...")
        
        # Initialize database
        db = VectorDatabase(DB_PATH)
        
        # STEP 1: Check if we have existing progress for embedding generation
        print("\n📊 STEP 1: Checking for existing progress...")
        
        # Get all operations to see if there's any ongoing embedding work
        operations = db.progress_tracker.get_all_operations()
        running_embedding_ops = [op for op in operations if op.status == 'running' and 'embedding' in op.operation_id]
        
        if running_embedding_ops:
            print(f"Found {len(running_embedding_ops)} running embedding operations:")
            for op in running_embedding_ops:
                print(f"   - {op.operation_id}: {op.processed_items}/{op.total_items} completed")
                print(f"   Resuming operation {op.operation_id}...")
                
                # Resume the existing operation
                result = await db.generate_missing_embeddings(op.operation_id)
                print(f"   ✅ Resumed and completed {result} more embeddings")
        else:
            print("No running embedding operations found.")
            
            # Check if we need to load data first
            stats = db.get_database_stats()
            total_records = stats.get('total_sub_intents', 0)
            
            if total_records == 0:
                print("\n📊 STEP 1a: Loading data from JSON file...")
                operation_id = f"json_load_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                success_count = await load_json_to_vector_db_with_progress(JSON_FILE_PATH, DB_PATH, operation_id)
                print(f"✅ Loaded {success_count} sub-intents into database")
            
            # Check for pending embeddings
            pending_records = db.get_pending_embeddings()
            
            if pending_records:
                print(f"\n📊 STEP 1b: Found {len(pending_records)} records needing embeddings")
                print("Starting embedding generation with progress tracking...")
                
                # Start new embedding generation
                embedding_operation_id = f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                embedding_success = await db.generate_missing_embeddings(embedding_operation_id, batch_size=5)
                print(f"✅ Generated {embedding_success} embeddings")
            else:
                print("✅ All records already have embeddings!")
        
        # STEP 2: Show final database statistics
        print("\n📊 STEP 2: Final Database Statistics")
        final_stats = db.get_database_stats()
        for key, value in final_stats.items():
            if key == 'progress_summary':
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            elif key != 'progress_summary':
                print(f"   {key}: {value}")
        
        # STEP 3: Test vector search with one query
        print("\n📊 STEP 3: Testing Vector Search")
        test_query = "Show me pending expenses for project dashboard"
        print(f"🔍 Search Query: '{test_query}'")
        
        try:
            results = await db.search_similar_sub_intents(test_query, confidence_threshold=0.6, limit=10)
            
            if results:
                print(f"✅ Found {len(results)} matching sub-intents:")
                for i, (record, confidence) in enumerate(results, 1):
                    print(f"   {i}. {record.name}")
                    print(f"      Confidence: {confidence:.3f}")
                    print(f"      Description: {record.description}")
                    print(f"      Keywords: {', '.join(record.keywords[:3])}{'...' if len(record.keywords) > 3 else ''}")
                    print(f"      Tables: {', '.join(record.tables[:3])}{'...' if len(record.tables) > 3 else ''}")
                    print()
            else:
                print("❌ No matches found above the confidence threshold")
                
                # Try with lower threshold
                print("🔍 Trying with lower confidence threshold (0.5)...")
                results_low = await db.search_similar_sub_intents(test_query, confidence_threshold=0.5, limit=3)
                
                if results_low:
                    print(f"✅ Found {len(results_low)} matches with lower threshold:")
                    for i, (record, confidence) in enumerate(results_low, 1):
                        print(f"   {i}. {record.name} (Confidence: {confidence:.3f})")
                else:
                    print("❌ No matches found even with lower threshold")
                        
        except Exception as e:
            print(f"❌ Error during search: {e}")
        
        # STEP 4: Show progress summary
        print("\n📊 STEP 4: Operations Summary")
        progress_summary = db.get_progress_summary()
        print("Recent Operations:")
        for op in progress_summary.get('operations', [])[:5]:  # Show top 5
            print(f"   - {op['operation_id']}")
            print(f"     Status: {op['status']} | Progress: {op['progress']} | Success Rate: {op['success_rate']}")
            if op['failed_items'] > 0:
                print(f"     Failed Items: {op['failed_items']}")
        
        print("\n✅ All operations completed successfully!")
        print("\n💡 To resume interrupted operations in the future, simply run this script again.")
        print("    The system will automatically detect and resume any incomplete work.")
    
    # Run the main function
    asyncio.run(main_with_progress_and_search())