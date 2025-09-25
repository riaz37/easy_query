
import os
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from contextlib import contextmanager

# Import your existing pipeline
from data_sources.file_data.app.unstructured.agent.data_structizer_agent_v2  import DocumentProcessingPipeline, FileMetadata, ChunkMetadata, create_pipeline
# Import the new config loader
from data_sources.file_data.app.unstructured.agent.config_loader import get_database_config, DatabaseConfig

# # Load environment variables
# load_dotenv(override=True)

# Retrieve API key and model name from environment variables
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentDatabase:
    """Handles all database operations for document storage."""
    
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
    
    def setup_database(self):
        """Create tables and install pgvector extension."""
        
        # SQL for creating tables
        create_extension_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;
        """
        
        create_chunks_table_sql = """
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id VARCHAR(50) PRIMARY KEY,
            file_id VARCHAR(36) NOT NULL REFERENCES document_files(file_id) ON DELETE CASCADE,
            chunk_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            title VARCHAR(1000) NOT NULL,
            keywords TEXT[], -- Array of keywords
            date_range JSONB, -- Store date range as JSON
            chunk_order INTEGER NOT NULL, -- Order within the document
            embedding vector(1536), -- Reduced to 1536 dimensions
            combined_context TEXT NOT NULL, 
            combined_embedding vector(1536), -- Reduced to 1536 dimensions
            metadata JSONB -- New column for file_name and page_numbers
        );
        """
        
        create_files_table_sql = """
        CREATE TABLE IF NOT EXISTS document_files (
            file_id VARCHAR(36) PRIMARY KEY,
            file_name VARCHAR(500) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            file_path TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            full_summary TEXT NOT NULL,
            title VARCHAR(1000) NOT NULL,
            keywords TEXT[], -- Array of keywords
            date_range JSONB, -- Store date range as JSON
            processing_timestamp TIMESTAMP NOT NULL,
            intent VARCHAR(500), -- For future use
            sub_intent VARCHAR(500), -- For future use
            title_summary_combined TEXT NOT NULL, 
            title_summary_embedding vector(1536), -- Reduced to 1536 dimensions for  embeddings
            file_description TEXT, -- New column for file description
            table_name VARCHAR(200), -- New column for table name
            user_id VARCHAR(100) -- New column for user id
        );
        """
        create_basic_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_files_file_id ON document_files(file_id);",
            "CREATE INDEX IF NOT EXISTS idx_files_keywords ON document_files USING GIN(keywords);",
            "CREATE INDEX IF NOT EXISTS idx_files_date_range ON document_files USING GIN(date_range);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON document_chunks(file_id);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_keywords ON document_chunks USING GIN(keywords);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_date_range ON document_chunks USING GIN(date_range);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON document_chunks USING GIN(metadata);",  # New index
        ]
        
        # Vector indexes - created separately with error handling
        vector_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_files_title_summary_embedding ON document_files USING hnsw (title_summary_embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_combined_embedding ON document_chunks USING hnsw (combined_embedding vector_cosine_ops);",
        ]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            try:
                # Create extension
                cursor.execute(create_extension_sql)
                logger.info("pgvector extension created/verified")
                
                # Create tables
                cursor.execute(create_files_table_sql)
                cursor.execute(create_chunks_table_sql)
                logger.info("Tables created successfully")
                
                # Create basic indexes
                for index_sql in create_basic_indexes_sql:
                    try:
                        cursor.execute(index_sql)
                        logger.info(f"Created basic index: {index_sql.split('idx_')[1].split(' ')[0]}")
                    except Exception as e:
                        logger.warning(f"Basic index creation warning: {e}")
                
                conn.commit()
                logger.info("Basic database setup completed successfully")
                
                # Create vector indexes separately (these might fail if no data exists yet)
                for index_sql in vector_indexes_sql:
                    try:
                        cursor.execute(index_sql)
                        logger.info(f"Created vector index: {index_sql.split('idx_')[1].split(' ')[0]}")
                        conn.commit()
                    except Exception as e:
                        conn.rollback()
                        logger.warning(f"Vector index creation skipped (will create after data insertion): {e}")
                
                logger.info("Database setup completed")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error setting up database: {e}")
                raise
    
    def create_vector_indexes_if_needed(self):
        """Create vector indexes after data has been inserted."""
        vector_indexes_sql = [
            "CREATE INDEX IF NOT EXISTS idx_files_title_summary_embedding ON document_files USING hnsw (title_summary_embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks USING hnsw (embedding vector_cosine_ops);",
            "CREATE INDEX IF NOT EXISTS idx_chunks_combined_embedding ON document_chunks USING hnsw (combined_embedding vector_cosine_ops);",
        ]
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for index_sql in vector_indexes_sql:
                try:
                    cursor.execute(index_sql)
                    conn.commit()
                    logger.info(f"Created vector index: {index_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"Vector index creation failed: {e}")
    
    def save_document_data(self, file_metadata: FileMetadata, chunk_metadata_list: List[ChunkMetadata]) -> bool:
        """Save file and chunk metadata to database without embeddings."""
        
        try:
            # Prepare file data with semantic formatting
            title_summary_combined = f"Title:\n{file_metadata.title}\n\nSummary:\n{file_metadata.full_summary}"
            
            # Prepare chunk data with semantic formatting
            combined_contexts = []
            for chunk in chunk_metadata_list:
                combined_context = f"Title:\n{file_metadata.title}\n\nFull Summary:\n{file_metadata.full_summary}\n\nChunk Summary:\n{chunk.summary}"
                combined_contexts.append(combined_context)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert file data
                file_insert_sql = """
                INSERT INTO document_files (
                    file_id, file_name, file_type, file_path, extracted_text, 
                    full_summary, title, keywords, date_range, processing_timestamp,
                    intent, sub_intent, title_summary_combined, title_summary_embedding,
                    file_description, table_name, user_id
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (file_id) DO UPDATE SET
                    file_name = EXCLUDED.file_name,
                    file_type = EXCLUDED.file_type,
                    file_path = EXCLUDED.file_path,
                    extracted_text = EXCLUDED.extracted_text,
                    full_summary = EXCLUDED.full_summary,
                    title = EXCLUDED.title,
                    keywords = EXCLUDED.keywords,
                    date_range = EXCLUDED.date_range,
                    processing_timestamp = EXCLUDED.processing_timestamp,
                    title_summary_combined = EXCLUDED.title_summary_combined,
                    file_description = EXCLUDED.file_description,
                    table_name = EXCLUDED.table_name,
                    user_id = EXCLUDED.user_id;
                """
                
                # Handle timestamp conversion
                try:
                    if isinstance(file_metadata.processing_timestamp, str):
                        if 'Z' in file_metadata.processing_timestamp:
                            timestamp = datetime.fromisoformat(file_metadata.processing_timestamp.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(file_metadata.processing_timestamp)
                    else:
                        timestamp = file_metadata.processing_timestamp
                except Exception as e:
                    logger.warning(f"Timestamp conversion issue: {e}, using current time")
                    timestamp = datetime.now()
                
                cursor.execute(file_insert_sql, (
                    file_metadata.file_id,
                    file_metadata.file_name,
                    file_metadata.file_type,
                    file_metadata.file_path,
                    file_metadata.extracted_text,
                    file_metadata.full_summary,
                    file_metadata.title,
                    file_metadata.keywords,
                    json.dumps(file_metadata.date_range) if file_metadata.date_range else None,
                    timestamp,
                    None,  # intent (empty for now)
                    None,  # sub_intent (empty for now)
                    title_summary_combined,
                    None,   # title_summary_embedding (will be filled by separate embedding function)
                    file_metadata.file_description,
                    file_metadata.table_name,
                    file_metadata.user_id
                ))
                
                logger.info(f"File metadata saved to database - file_description: '{file_metadata.file_description}', table_name: '{file_metadata.table_name}', user_id: '{file_metadata.user_id}'")
                logger.info("File metadata saved to database")
                
                # Delete existing chunks for this file (in case of re-processing)
                cursor.execute("DELETE FROM document_chunks WHERE file_id = %s", (file_metadata.file_id,))
                
                # Insert chunk data
                chunk_insert_sql = """
                INSERT INTO document_chunks (
                    chunk_id, file_id, chunk_text, summary, title, keywords, 
                    date_range, chunk_order, embedding, combined_context, combined_embedding, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                );
                """
                print("Inserting chunk data into database...")
                logger.info(f"Inserting {len(chunk_metadata_list)} chunks for file {file_metadata.file_id}")
                for i, chunk in enumerate(chunk_metadata_list):
                    combined_context = combined_contexts[i]
                    
                    # Create metadata JSON
                    metadata_json = {
                        "file_name": file_metadata.file_name,
                        "page_numbers": chunk.page_numbers if hasattr(chunk, 'page_numbers') else []
                    }
                    
                    cursor.execute(chunk_insert_sql, (
                        chunk.chunk_id,
                        chunk.file_id,
                        chunk.chunk_text,
                        chunk.summary,
                        chunk.title,
                        chunk.keywords,
                        json.dumps(chunk.date_range) if chunk.date_range else None,
                        i + 1,  # chunk_order
                        None,   # embedding (will be filled by separate embedding function)
                        combined_context,
                        None,   # combined_embedding (will be filled by separate embedding function)
                        json.dumps(metadata_json)  # metadata
                    ))
                
                conn.commit()
                logger.info(f"Successfully saved {len(chunk_metadata_list)} chunks to database")
                return True
                
        except Exception as e:
            logger.error(f"Error saving document data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_file_embedding(self, file_id: str, embedding: List[float]) -> bool:
        """Update file embedding in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE document_files SET title_summary_embedding = %s WHERE file_id = %s",
                    (embedding, file_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating file embedding: {e}")
            return False
    
    def update_chunk_embeddings(self, chunk_id: str, text_embedding: List[float], combined_embedding: List[float]) -> bool:
        """Update chunk embeddings in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE document_chunks SET embedding = %s, combined_embedding = %s WHERE chunk_id = %s",
                    (text_embedding, combined_embedding, chunk_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating chunk embeddings: {e}")
            return False
    
    def get_file_data_for_embedding(self, file_id: str) -> Optional[str]:
        """Get title_summary_combined text for embedding generation."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT title_summary_combined FROM document_files WHERE file_id = %s",
                    (file_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting file data for embedding: {e}")
            return None
    
    def get_chunk_data_for_embedding(self, chunk_id: str) -> Optional[Tuple[str, str]]:
        """Get chunk_text and combined_context for embedding generation."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT chunk_text, combined_context FROM document_chunks WHERE chunk_id = %s",
                    (chunk_id,)
                )
                result = cursor.fetchone()
                return result if result else None
        except Exception as e:
            logger.error(f"Error getting chunk data for embedding: {e}")
            return None

class DocumentDatabasePipeline:
    """Main pipeline that combines document processing with database storage."""
    
    def __init__(self, 
                 document_processor,
                 text_chunker, 
                 llm_client,
                 db_config: DatabaseConfig,
                 max_workers: int = 4):
        
        self.processing_pipeline = create_pipeline(document_processor, text_chunker, llm_client, max_workers)
        self.database = DocumentDatabase(db_config)
        
    def setup(self):
        """Setup database tables and indexes."""
        self.database.setup_database()
        
    def process_and_store_document(self, file_path: str, output_dir: Optional[str] = None, 
                                  file_description: Optional[str] = None, 
                                  table_name: Optional[str] = None,
                                  user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process document and store in database.
        
        Args:
            file_path: Path to the document to process
            output_dir: Optional directory to save chunks
            file_description: Optional description for the file
            table_name: Optional table name for the file
            user_id: Optional user id for the file
        
        Returns:
            Dict with processing results and database status
        """
        
        try:
            logger.info(f"DEBUG: Passing user_id={user_id} to process_document")
            # Step 1: Process document using existing pipeline
            logger.info(f"Processing document: {file_path}")
            file_metadata, chunk_metadata_list = self.processing_pipeline.process_document(
                file_path, output_dir, file_description, table_name, user_id=user_id
            )
            
            # Step 2: Save to database
            logger.info("Saving to database...")
            db_success = self.database.save_document_data(file_metadata, chunk_metadata_list)
            
            # Step 3: Create vector indexes if this is the first document
            if db_success:
                self.database.create_vector_indexes_if_needed()
            
            # Step 4: Optionally save to files as well
            saved_files = {}
            if output_dir:
                saved_files = self.processing_pipeline.save_results(file_metadata, chunk_metadata_list, output_dir)
            
            return {
                "success": True,
                "file_metadata": file_metadata,
                "chunk_count": len(chunk_metadata_list),
                "database_saved": db_success,
                "files_saved": saved_files,
                "file_id": file_metadata.file_id
            }
            
        except Exception as e:
            logger.error(f"Error in process_and_store_document: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_document_count(self) -> Dict[str, int]:
        """Get count of documents and chunks in database."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM document_files")
                file_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                chunk_count = cursor.fetchone()[0]
                
                return {
                    "file_count": file_count,
                    "chunk_count": chunk_count
                }
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return {"file_count": 0, "chunk_count": 0}
    
    def get_files_without_embeddings(self) -> List[str]:
        """Get list of file_ids that don't have embeddings yet."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT file_id FROM document_files WHERE title_summary_embedding IS NULL")
                file_ids = [row[0] for row in cursor.fetchall()]
                
                return file_ids
        except Exception as e:
            logger.error(f"Error getting files without embeddings: {e}")
            return []
    
    def get_chunks_without_embeddings(self) -> List[str]:
        """Get list of chunk_ids that don't have embeddings yet."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT chunk_id FROM document_chunks WHERE embedding IS NULL OR combined_embedding IS NULL")
                chunk_ids = [row[0] for row in cursor.fetchall()]
                
                return chunk_ids
        except Exception as e:
            logger.error(f"Error getting chunks without embeddings: {e}")
            return []
    
    def search_similar_documents(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                search_sql = """
                SELECT 
                    file_id, file_name, title, full_summary, keywords,
                    1 - (title_summary_embedding <=> %s) as similarity
                FROM document_files 
                WHERE title_summary_embedding IS NOT NULL
                ORDER BY title_summary_embedding <=> %s
                LIMIT %s
                """
                
                cursor.execute(search_sql, (query_embedding, query_embedding, limit))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
            return []
    
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 10, use_combined: bool = True) -> List[Dict]:
        """Search for similar chunks using cosine similarity."""
        try:
            with self.database.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                embedding_column = "combined_embedding" if use_combined else "embedding"
                
                search_sql = f"""
                SELECT 
                    c.chunk_id, c.file_id, c.chunk_text, c.summary, c.title, c.keywords,
                    c.metadata, -- Add metadata column
                    f.file_name, f.title as document_title,
                    1 - (c.{embedding_column} <=> %s) as similarity
                FROM document_chunks c
                JOIN document_files f ON c.file_id = f.file_id
                WHERE c.{embedding_column} IS NOT NULL
                ORDER BY c.{embedding_column} <=> %s
                LIMIT %s
"""
                
                cursor.execute(search_sql, (query_embedding, query_embedding, limit))
                results = cursor.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []

import os
from dotenv import load_dotenv
import logging
from data_sources.file_data.app.unstructured.agent.data_structizer_agent_v2  import SmartDocumentProcessor, SmartTextChunker, llm

# Load .env values
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_save(
    file_path: str,
    max_pages_per_chunk: int = 5,
    boundary_sentences: int = 3,
    boundary_table_rows: int = 3,
    target_pages_per_chunk: int = 3,
    overlap_pages: int = 1,
    min_pages_per_chunk: int = 1,
    respect_boundaries: bool = True,
    max_workers: int = 4,
    file_description: Optional[str] = None,  # New parameter
    table_name: Optional[str] = None,        # New parameter
    user_id: Optional[str] = None            # New parameter
):
    """Run the document processing pipeline with configurable parameters."""

    # Database config from API or environment variables
    db_config = get_database_config(user_id)

    try:
        # Initialize components
        document_processor = SmartDocumentProcessor(
            max_pages_per_chunk=max_pages_per_chunk,
            boundary_sentences=boundary_sentences,
            boundary_table_rows=boundary_table_rows
        )

        text_chunker = SmartTextChunker(
            target_pages_per_chunk=target_pages_per_chunk,
            overlap_pages=overlap_pages,
            max_pages_per_chunk=max_pages_per_chunk,
            min_pages_per_chunk=min_pages_per_chunk,
            respect_boundaries=respect_boundaries
        )

        # Create pipeline
        pipeline = DocumentDatabasePipeline(
            document_processor=document_processor,
            text_chunker=text_chunker,
            llm_client=llm,
            db_config=db_config,
            max_workers=max_workers
        )

        # Setup database
        pipeline.setup()

        # Process and store document
        result = pipeline.process_and_store_document(
            file_path, "processing_results", file_description, table_name, user_id
        )

        if result["success"]:
            print(f"✅ Document processed successfully!")
            print(f"File ID: {result['file_id']}")
            print(f"Chunks created: {result['chunk_count']}")
            print(f"Database saved: {result['database_saved']}")

            # Check database status
            counts = pipeline.get_document_count()
            print(f"Total files in database: {counts['file_count']}")
            print(f"Total chunks in database: {counts['chunk_count']}")

            files_without_embeddings = pipeline.get_files_without_embeddings()
            chunks_without_embeddings = pipeline.get_chunks_without_embeddings()

            print(f"Files without embeddings: {len(files_without_embeddings)}")
            print(f"Chunks without embeddings: {len(chunks_without_embeddings)}")

        else:
            print(f"❌ Error: {result['error']}")

    except Exception as e:
        logger.error(f"Error in data_save: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    data_save(file_path=r"/Users/nilab/Desktop/projects/Knowladge-Base/app/Agronochain Tech Doc.pdf")
