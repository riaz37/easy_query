import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import re
from contextlib import contextmanager
import google.generativeai as genai
import numpy as np

# Import the new config loader
from data_sources.file_data.app.unstructured.agent.config_loader import get_database_config, DatabaseConfig

# Load environment variables
load_dotenv(override=True)

# Retrieve API key and model name from environment variables
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")

gemini_embedding_model_name = os.getenv("google_gemini_embedding_name", "gemini-embedding-exp-03-07")
if gemini_embedding_model_name is None:
    print("Warning: 'google_gemini_embedding_name' not found in environment variables, using default 'gemini-embedding-exp-03-07'.")
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    api_key: str
    model: str = gemini_embedding_model_name
    batch_size: int = 50
    delay_between_requests: float = 2.0  # Delay in seconds between requests to avoid rate limiting
    max_retries: int = 3
    retry_delay: float = 2.0
    output_dimensionality: int = 1536
    
    file_embedding_task: str = "RETRIEVAL_DOCUMENT"
    chunk_embedding_task: str = "RETRIEVAL_DOCUMENT" 
    combined_embedding_task: str = "RETRIEVAL_DOCUMENT"



@dataclass
class ProcessingStats:
    """Simple processing statistics."""
    files_total: int = 0
    files_without_embeddings: int = 0
    files_processed: int = 0
    files_failed: int = 0
    
    chunks_total: int = 0
    chunks_without_embeddings: int = 0
    chunks_processed: int = 0
    chunks_failed: int = 0
    
    start_time: Optional[datetime] = None
    
    def get_files_remaining(self) -> int:
        return self.files_without_embeddings - self.files_processed - self.files_failed
    
    def get_chunks_remaining(self) -> int:
        return self.chunks_without_embeddings - self.chunks_processed - self.chunks_failed

class TextProcessor:
    """Handles text preprocessing for better embeddings."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for consistent embeddings."""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def prepare_file_text(title: str, summary: str) -> str:
        """Prepare file-level text for embedding with semantic structure."""
        normalized_title = TextProcessor.normalize_text(title)
        normalized_summary = TextProcessor.normalize_text(summary)
        
        combined_text = f"document title: {normalized_title}\n\ndocument summary: {normalized_summary}"
        
        return combined_text
    
    @staticmethod
    def prepare_chunk_text(chunk_text: str) -> str:
        """Prepare chunk text for embedding."""
        normalized_text = TextProcessor.normalize_text(chunk_text)
        return normalized_text
    
    @staticmethod
    def prepare_combined_text(title: str, full_summary: str, chunk_summary: str) -> str:
        """Prepare combined context text for embedding."""
        normalized_title = TextProcessor.normalize_text(title)
        normalized_full_summary = TextProcessor.normalize_text(full_summary)
        normalized_chunk_summary = TextProcessor.normalize_text(chunk_summary)
        
        combined_text = f"document title: {normalized_title}\n\ndocument overview: {normalized_full_summary}\n\nchunk overview: {normalized_chunk_summary}"
        
        return combined_text

class GeminiEmbeddingClient:
    """Handles Gemini API calls for embedding generation."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model)
        
    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT", output_dimensionality: int = 1536) -> Optional[List[float]]:
        """Generate embedding for a single text."""
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                
                result = genai.embed_content(
                    model=self.config.model,
                    content=text,
                    task_type=task_type,
                    output_dimensionality=output_dimensionality
                )
                
                await asyncio.sleep(self.config.delay_between_requests)
                
                return result['embedding']
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for embedding generation: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts: {e}")
                    return None
                    
        return None
    
    async def generate_embeddings_batch(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT", output_dimensionality: int = 1536) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        
        for i, text in enumerate(texts):
            embedding = await self.generate_embedding(text, task_type, output_dimensionality)
            embeddings.append(embedding)
        
        return embeddings

class EmbeddingDatabase:
    """Handles database operations for embeddings."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def get_embedding_counts(self) -> Dict[str, int]:
        """Get counts of files and chunks with/without embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # File counts
                cursor.execute("SELECT COUNT(*) FROM document_files")
                files_total = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_files WHERE title_summary_embedding IS NULL")
                files_without_embeddings = cursor.fetchone()[0]
                
                # Chunk counts
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                chunks_total = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE embedding IS NULL OR combined_embedding IS NULL")
                chunks_without_embeddings = cursor.fetchone()[0]
                
                return {
                    "files_total": files_total,
                    "files_without_embeddings": files_without_embeddings,
                    "files_with_embeddings": files_total - files_without_embeddings,
                    "chunks_total": chunks_total,
                    "chunks_without_embeddings": chunks_without_embeddings,
                    "chunks_with_embeddings": chunks_total - chunks_without_embeddings
                }
                
        except Exception as e:
            logger.error(f"Error getting embedding counts: {e}")
            return {}
    
    def get_files_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get files that need embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT file_id, title, full_summary, title_summary_combined
                    FROM document_files 
                    WHERE title_summary_embedding IS NULL
                    ORDER BY processing_timestamp DESC
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting files without embeddings: {e}")
            return []
    
    def get_chunks_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get chunks that need embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT c.chunk_id, c.file_id, c.chunk_text, c.summary, c.combined_context,
                           f.title, f.full_summary
                    FROM document_chunks c
                    JOIN document_files f ON c.file_id = f.file_id
                    WHERE c.embedding IS NULL OR c.combined_embedding IS NULL
                    ORDER BY c.file_id, c.chunk_order
                """)
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting chunks without embeddings: {e}")
            return []
    
    def update_file_embedding(self, file_id: str, embedding: List[float]) -> bool:
        """Update file embedding in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE document_files 
                    SET title_summary_embedding = %s 
                    WHERE file_id = %s
                """, (embedding, file_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating file embedding for {file_id}: {e}")
            return False
    
    def update_chunk_embeddings(self, chunk_id: str, chunk_embedding: List[float], combined_embedding: List[float]) -> bool:
        """Update chunk embeddings in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE document_chunks 
                    SET embedding = %s, combined_embedding = %s 
                    WHERE chunk_id = %s
                """, (chunk_embedding, combined_embedding, chunk_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error updating chunk embeddings for {chunk_id}: {e}")
            return False

class GeminiEmbeddingManager:
    """Main class for managing Gemini embeddings with clear progress tracking."""
    
    def __init__(self, 
                 embedding_config: EmbeddingConfig,
                 db_config: DatabaseConfig):
        
        self.embedding_config = embedding_config
        self.db_config = db_config
        self.client = GeminiEmbeddingClient(embedding_config)
        self.database = EmbeddingDatabase(db_config)
        self.text_processor = TextProcessor()
        self.stats = ProcessingStats()
    
    def display_initial_status(self):
        """Display initial status of embeddings needed."""
        counts = self.database.get_embedding_counts()
        
        self.stats.files_total = counts.get("files_total", 0)
        self.stats.files_without_embeddings = counts.get("files_without_embeddings", 0)
        self.stats.chunks_total = counts.get("chunks_total", 0)
        self.stats.chunks_without_embeddings = counts.get("chunks_without_embeddings", 0)
        
        print("\n" + "="*60)
        print("📊 EMBEDDING STATUS CHECK")
        print("="*60)
        print(f"📄 FILES:")
        print(f"   Total files in database: {self.stats.files_total}")
        print(f"   Files with embeddings: {counts.get('files_with_embeddings', 0)}")
        print(f"   Files WITHOUT embeddings: {self.stats.files_without_embeddings}")
        print()
        print(f"🧩 CHUNKS:")
        print(f"   Total chunks in database: {self.stats.chunks_total}")
        print(f"   Chunks with embeddings: {counts.get('chunks_with_embeddings', 0)}")
        print(f"   Chunks WITHOUT embeddings: {self.stats.chunks_without_embeddings}")
        print("="*60)
        
        if self.stats.files_without_embeddings == 0 and self.stats.chunks_without_embeddings == 0:
            print("✅ All files and chunks already have embeddings!")
            return False
        else:
            print(f"🚀 Ready to process {self.stats.files_without_embeddings} files and {self.stats.chunks_without_embeddings} chunks")
            return True
    
    def display_progress(self, item_type: str):
        """Display current progress."""
        if item_type == "files":
            processed = self.stats.files_processed
            failed = self.stats.files_failed
            remaining = self.stats.get_files_remaining()
            total_needed = self.stats.files_without_embeddings
        else:
            processed = self.stats.chunks_processed
            failed = self.stats.chunks_failed
            remaining = self.stats.get_chunks_remaining()
            total_needed = self.stats.chunks_without_embeddings
        
        if total_needed > 0:
            progress_pct = (processed / total_needed) * 100
            print(f"📈 {item_type.upper()} PROGRESS: {processed}/{total_needed} ({progress_pct:.1f}%) | Remaining: {remaining} | Failed: {failed}")
    
    async def process_file_embeddings(self):
        """Process embeddings for files."""
        if self.stats.files_without_embeddings == 0:
            print("ℹ️  No files need embeddings - skipping")
            return
        
        print(f"\n🔄 Processing embeddings for {self.stats.files_without_embeddings} files...")
        
        files_to_process = self.database.get_files_without_embeddings()
        
        for i in range(0, len(files_to_process), self.embedding_config.batch_size):
            batch = files_to_process[i:i + self.embedding_config.batch_size]
            
            # Prepare texts for embedding
            texts = []
            for file_data in batch:
                text = self.text_processor.prepare_file_text(
                    file_data['title'], 
                    file_data['full_summary']
                )
                texts.append(text)
            
            # Generate embeddings
            embeddings = await self.client.generate_embeddings_batch(
                texts, 
                self.embedding_config.file_embedding_task,
                output_dimensionality=self.embedding_config.output_dimensionality
            )
            
            # Update database
            for file_data, embedding in zip(batch, embeddings):
                if embedding:
                    success = self.database.update_file_embedding(file_data['file_id'], embedding)
                    if success:
                        self.stats.files_processed += 1
                    else:
                        self.stats.files_failed += 1
                else:
                    self.stats.files_failed += 1
            
            # Show progress
            self.display_progress("files")
    
    async def process_chunk_embeddings(self):
        """Process embeddings for chunks."""
        if self.stats.chunks_without_embeddings == 0:
            print("ℹ️  No chunks need embeddings - skipping")
            return
        
        print(f"\n🔄 Processing embeddings for {self.stats.chunks_without_embeddings} chunks...")
        
        chunks_to_process = self.database.get_chunks_without_embeddings()
        
        for i in range(0, len(chunks_to_process), self.embedding_config.batch_size):
            batch = chunks_to_process[i:i + self.embedding_config.batch_size]
            
            # Prepare texts for embedding
            chunk_texts = []
            combined_texts = []
            
            for chunk_data in batch:
                chunk_text = self.text_processor.prepare_chunk_text(chunk_data['summary'])
                chunk_texts.append(chunk_text)
                
                combined_text = self.text_processor.prepare_combined_text(
                    chunk_data['title'],
                    chunk_data['full_summary'],
                    chunk_data['summary']
                )
                combined_texts.append(combined_text)
            
            # Generate embeddings
            chunk_embeddings = await self.client.generate_embeddings_batch(
                chunk_texts, 
                self.embedding_config.chunk_embedding_task,
                output_dimensionality=self.embedding_config.output_dimensionality
            )
            
            combined_embeddings = await self.client.generate_embeddings_batch(
                combined_texts, 
                self.embedding_config.combined_embedding_task,
                output_dimensionality=self.embedding_config.output_dimensionality
            )
            
            # Update database
            for chunk_data, chunk_embedding, combined_embedding in zip(batch, chunk_embeddings, combined_embeddings):
                if chunk_embedding and combined_embedding:
                    success = self.database.update_chunk_embeddings(
                        chunk_data['chunk_id'], 
                        chunk_embedding, 
                        combined_embedding
                    )
                    if success:
                        self.stats.chunks_processed += 1
                    else:
                        self.stats.chunks_failed += 1
                else:
                    self.stats.chunks_failed += 1
            
            # Show progress
            self.display_progress("chunks")
    
    async def process_all_embeddings(self):
        """Process all embeddings with clear progress tracking."""
        # Check initial status
        needs_processing = self.display_initial_status()
        
        if not needs_processing:
            return
        
        self.stats.start_time = datetime.now()
        
        # Process files first
        await self.process_file_embeddings()
        
        # Process chunks
        await self.process_chunk_embeddings()
        
        # Final summary
        self.display_final_summary()
    
    def display_final_summary(self):
        """Display final processing summary."""
        elapsed_time = datetime.now() - self.stats.start_time if self.stats.start_time else None
        elapsed_str = f" in {elapsed_time}" if elapsed_time else ""
        
        print("\n" + "="*60)
        print("🎉 EMBEDDING PROCESSING COMPLETED")
        print("="*60)
        print(f"📄 FILES:")
        print(f"   ✅ Successfully processed: {self.stats.files_processed}")
        print(f"   ❌ Failed: {self.stats.files_failed}")
        print(f"   📋 Remaining: {self.stats.get_files_remaining()}")
        print()
        print(f"🧩 CHUNKS:")
        print(f"   ✅ Successfully processed: {self.stats.chunks_processed}")
        print(f"   ❌ Failed: {self.stats.chunks_failed}")
        print(f"   📋 Remaining: {self.stats.get_chunks_remaining()}")
        print()
        print(f"⏱️  Total time{elapsed_str}")
        print("="*60)

# Main execution function
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv(override=True)

async def embed_gen(
    batch_size: int = 20,
    delay_between_requests: int = 2,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    user_id: Optional[str] = None
):
    """Main function to run embedding processing."""
    # API Key from .env
    gemini_apikey = os.getenv("google_api_key")
    if gemini_apikey is None:
        print("❌ Error: 'google_api_key' not found in environment variables.")
        return

    # Configuration
    embedding_config = EmbeddingConfig(
        api_key=gemini_apikey,
        model=gemini_embedding_model_name,
        batch_size=batch_size,
        delay_between_requests=delay_between_requests,
        max_retries=max_retries,
        retry_delay=retry_delay
    )

    # Load database config from API or environment variables
    db_config = get_database_config(user_id)

    # Initialize manager
    manager = GeminiEmbeddingManager(
        embedding_config=embedding_config,
        db_config=db_config
    )

    try:
        print("🚀 Starting Gemini Embedding Processing...")

        # Process all embeddings
        await manager.process_all_embeddings()

        print("✅ Embedding processing completed successfully!")

    except Exception as e:
        print(f"❌ Error in embedding processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Example call with your desired model name
    asyncio.run(embed_gen(
        batch_size=20,
        delay_between_requests=2,
        max_retries=3,
        retry_delay=2.0
    ))
