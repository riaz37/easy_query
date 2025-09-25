import os
import json
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, List, Any, Optional
import logging

# Import the centralized configuration system
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'unstructured', 'agent'))
from config_loader import DatabaseConfig, get_database_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkDataInserter:
    """Handles insertion of chunk data from JSON to database."""
    
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
    
    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read JSON data from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Successfully read JSON file: {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise
    
    def extract_keywords_from_chunks(self, chunks: List[Dict]) -> List[str]:
        """Extract all keywords from chunks."""
        all_keywords = []
        for chunk in chunks:
            table_analysis = chunk.get("table_analysis", {})
            if table_analysis and "tables" in table_analysis:
                for table in table_analysis["tables"]:
                    keywords = table.get("keywords", [])
                    all_keywords.extend(keywords)
        
        # Remove duplicates and return
        return list(set(all_keywords))
    
    def extract_date_range_from_chunks(self, chunks: List[Dict]) -> Dict[str, str]:
        """Extract date range from chunks."""
        for chunk in chunks:
            table_analysis = chunk.get("table_analysis", {})
            if table_analysis and "tables" in table_analysis:
                for table in table_analysis["tables"]:
                    date_range = table.get("date_range", {})
                    if date_range:
                        return date_range
        
        # Default date range if none found
        return {
            "end": "2021-06-30",
            "start": "2019-07-01"
        }
    
    def extract_column_names_from_chunks(self, chunks: List[Dict]) -> List[str]:
        """Extract all column names from chunks."""
        all_columns = []
        for chunk in chunks:
            table_analysis = chunk.get("table_analysis", {})
            if table_analysis and "tables" in table_analysis:
                for table in table_analysis["tables"]:
                    column_names = table.get("column_names", [])
                    all_columns.extend(column_names)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in all_columns:
            if col and col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        return unique_columns
    
    def create_chunk_text_with_columns(self, chunk: Dict) -> str:
        """Create chunk text with column names prepended for individual chunk."""
        chunk_text = ""
        
        # Extract column names for this specific chunk
        table_analysis = chunk.get("table_analysis", {})
        column_names = []
        if table_analysis and "tables" in table_analysis:
            for table in table_analysis["tables"]:
                chunk_column_names = table.get("column_names", [])
                column_names.extend(chunk_column_names)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in column_names:
            if col and col not in seen:
                seen.add(col)
                unique_columns.append(col)
        
        # Add column names at the beginning if they exist
        if unique_columns:
            chunk_text += "Column Names:\n"
            chunk_text += ", ".join(unique_columns) + "\n\n"
        
        # Add the original chunk text
        original_chunk_text = chunk.get("chunk_text", "")
        if original_chunk_text:
            chunk_text += original_chunk_text
        
        return chunk_text.strip()
    
    def create_extracted_text_with_columns(self, chunks: List[Dict]) -> str:
        """Create extracted text with column names included."""
        extracted_text = ""
        
        # Add column names at the beginning
        column_names = self.extract_column_names_from_chunks(chunks)
        if column_names:
            extracted_text += "Column Names:\n"
            extracted_text += ", ".join(column_names) + "\n\n"
        
        # Add all chunk texts
        for chunk in chunks:
            chunk_text = chunk.get("chunk_text", "")
            if chunk_text:
                extracted_text += chunk_text + "\n\n"
        
        return extracted_text.strip()
    
    def insert_file_and_chunks_from_json(self, json_file_path: str, file_name: str, file_description: Optional[str] = None, table_name: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """
        Insert file and chunk data from JSON file into database.
        
        Args:
            json_file_path: Path to the JSON file
            file_name: Name of the file to use in metadata
            file_description: Optional description for the file
            table_name: Optional table name for the file
            user_id: Optional user ID for the file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read JSON data
            data = self.read_json_file(json_file_path)
            
            # Extract file-level information
            file_id = data.get("file_id")
            combined_title = data.get("combined_title", "")
            combined_description = data.get("combined_description", "")
            document_summary = data.get("combined_description", "")
            processed_at = data.get("processed_at", "")
            file_path = data.get("file_path", "")
            chunks = data.get("chunks", [])
            total_chunks = data.get("total_chunks", len(chunks))
            
            if not file_id:
                logger.error("No file_id found in JSON data")
                return False
            
            # Extract keywords from chunks
            keywords = self.extract_keywords_from_chunks(chunks)
            
            # Extract date range from chunks
            date_range = self.extract_date_range_from_chunks(chunks)
            
            # Create combined context base (title + description)
            combined_context_base = f"{combined_title}\n\n{combined_description}"
            
            # Create extracted text with column names
            extracted_text = self.create_extracted_text_with_columns(chunks)
            
            # Insert file and chunks into database
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # First, insert or update the file record
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
                
                # Handle timestamp - use current time if not provided
                from datetime import datetime
                if processed_at:
                    try:
                        if 'T' in processed_at:
                            timestamp = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.fromisoformat(processed_at)
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                # Set file type as xlsx
                file_type = "xlsx"
                
                # Create title_summary_combined for the file
                title_summary_combined = f"Title:\n{combined_title}\n\nSummary:\n{combined_description}"
                
                # Insert file record
                cursor.execute(file_insert_sql, (
                    file_id,                     # file_id
                    file_name,                   # file_name
                    file_type,                   # file_type
                    file_path,                   # file_path
                    extracted_text,              # extracted_text
                    combined_description,        # full_summary (using combined_description)
                    combined_title,              # title
                    keywords,                    # keywords
                    json.dumps(date_range),      # date_range
                    timestamp,                   # processing_timestamp
                    None,                        # intent
                    None,                        # sub_intent
                    title_summary_combined,      # title_summary_combined
                    None,                        # title_summary_embedding
                    file_description,            # file_description
                    table_name,                  # table_name
                    user_id                      # user_id
                ))
                
                logger.info(f"Inserted/Updated file record for file_id: {file_id}")
                
                # Delete existing chunks for this file (if any)
                cursor.execute("DELETE FROM document_chunks WHERE file_id = %s", (file_id,))
                logger.info(f"Deleted existing chunks for file_id: {file_id}")
                
                # Insert chunks if they exist
                if chunks:
                    chunk_insert_sql = """
                    INSERT INTO document_chunks (
                        chunk_id, file_id, chunk_text, summary, title, keywords, 
                        date_range, chunk_order, embedding, combined_context, combined_embedding, metadata
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                    """
                    
                    successfully_inserted = 0
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            # Extract chunk data
                            chunk_id = chunk.get("chunk_id")
                            table_analysis = chunk.get("table_analysis", {})
                            
                            # FIXED MAPPING:
                            # 1. title -> Use chunk's "title" field (chunk title)
                            chunk_title = chunk.get("title", "")
                            
                            # 2. summary -> Use table_description from table_analysis
                            table_description = ""
                            if table_analysis and "tables" in table_analysis:
                                for table in table_analysis["tables"]:
                                    if table.get("table_description"):
                                        table_description = table["table_description"]
                                        break
                            
                            # If no table_description found, use chunk title as fallback
                            if not table_description:
                                table_description = chunk_title
                            
                            # 3. chunk_text -> Add column names BEFORE the original chunk_text
                            chunk_text_with_columns = self.create_chunk_text_with_columns(chunk)
                            
                            # Extract keywords for this chunk
                            chunk_keywords = []
                            if table_analysis and "tables" in table_analysis:
                                for table in table_analysis["tables"]:
                                    chunk_keywords.extend(table.get("keywords", []))
                            
                            # Remove duplicates
                            chunk_keywords = list(set(chunk_keywords))
                            
                            # Extract date range for this chunk
                            chunk_date_range = date_range  # Use file-level date range as default
                            if table_analysis and "tables" in table_analysis:
                                for table in table_analysis["tables"]:
                                    if table.get("date_range"):
                                        chunk_date_range = table["date_range"]
                                        break
                            
                            # Create combined context
                            table_descriptions = []
                            if table_analysis and "tables" in table_analysis:
                                for table in table_analysis["tables"]:
                                    if table.get("table_description"):
                                        table_descriptions.append(table["table_description"])
                            
                            combined_context = combined_context_base
                            if table_descriptions:
                                combined_context += "\n\nTable Descriptions:\n" + "\n".join(table_descriptions)
                            
                            # Create metadata with additional chunk info
                            metadata = {
                                "file_name": file_name,
                                "page_numbers": [1],  # Default page number
                                "chunk_size": chunk.get("chunk_size", 0),
                                "table_group_id": chunk.get("table_group_id", "")
                            }
                            
                            # Generate chunk_id if not present
                            if not chunk_id:
                                chunk_id = str(uuid.uuid4())
                            
                            # Insert the chunk with corrected field mapping
                            cursor.execute(chunk_insert_sql, (
                                chunk_id,                         # chunk_id
                                file_id,                         # file_id
                                chunk_text_with_columns,         # chunk_text (with column names prepended)
                                table_description,               # summary (table_description from table_analysis)
                                chunk_title,                     # title (chunk's title field)
                                chunk_keywords,                  # keywords (chunk-specific)
                                json.dumps(chunk_date_range),    # date_range (chunk-specific)
                                i + 1,                          # chunk_order
                                None,                           # embedding (NULL for now)
                                combined_context,               # combined_context
                                None,                           # combined_embedding (NULL for now)
                                json.dumps(metadata)            # metadata
                            ))
                            
                            successfully_inserted += 1
                            logger.info(f"Inserted chunk {i + 1}/{len(chunks)}: {chunk_id}")
                            logger.info(f"  - Title: {chunk_title}")
                            logger.info(f"  - Summary: {table_description[:100]}...")
                            logger.info(f"  - Chunk text length: {len(chunk_text_with_columns)}")
                            
                        except Exception as e:
                            logger.error(f"Error inserting chunk {i + 1}: {e}")
                            # Don't continue with other chunks if one fails in the same transaction
                            conn.rollback()
                            raise e
                    
                    logger.info(f"Successfully inserted {successfully_inserted}/{len(chunks)} chunks")
                
                conn.commit()
                logger.info("✅ All data committed successfully")
                
                return True
                
        except Exception as e:
            logger.error(f"Error in insert_file_and_chunks_from_json: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_chunk_count(self, file_id: str) -> int:
        """Get count of chunks for a specific file."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE file_id = %s", (file_id,))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting chunk count: {e}")
            return 0
    
    def validate_json_fields(self, json_file_path: str) -> Dict[str, bool]:
        """Validate that all expected JSON fields are accessible."""
        try:
            data = self.read_json_file(json_file_path)
            
            # File-level fields
            file_fields = {
                "file_id": data.get("file_id") is not None,
                "file_path": data.get("file_path") is not None,
                "combined_title": data.get("combined_title") is not None,
                "combined_description": data.get("combined_description") is not None,
                "total_chunks": data.get("total_chunks") is not None,
                "processed_at": data.get("processed_at") is not None,
                "chunks": data.get("chunks") is not None,
                "document_summary": data.get("document_summary") is not None,
            }
            
            # Chunk-level fields (check first chunk if exists)
            chunk_fields = {}
            chunks = data.get("chunks", [])
            if chunks:
                first_chunk = chunks[0]
                chunk_fields = {
                    "chunk_title": first_chunk.get("title") is not None,
                    "chunk_id": first_chunk.get("chunk_id") is not None,
                    "chunk_text": first_chunk.get("chunk_text") is not None,
                    "chunk_size": first_chunk.get("chunk_size") is not None,
                    "table_group_id": first_chunk.get("table_group_id") is not None,
                    "table_analysis": first_chunk.get("table_analysis") is not None,
                }
                
                # Table analysis fields
                table_analysis = first_chunk.get("table_analysis", {})
                if table_analysis and "tables" in table_analysis:
                    tables = table_analysis["tables"]
                    if tables:
                        first_table = tables[0]
                        table_fields = {
                            "table_description": first_table.get("table_description") is not None,
                            "keywords": first_table.get("keywords") is not None,
                            "date_range": first_table.get("date_range") is not None,
                            "column_names": first_table.get("column_names") is not None,
                            "last_lines_for_chunking": first_table.get("last_lines_for_chunking") is not None,
                        }
                        chunk_fields.update(table_fields)
            
            validation_result = {**file_fields, **chunk_fields}
            
            # Log validation results
            logger.info("JSON Field Validation Results:")
            for field, is_accessible in validation_result.items():
                status = "✅ ACCESSIBLE" if is_accessible else "❌ NOT ACCESSIBLE"
                logger.info(f"  {field}: {status}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating JSON fields: {e}")
            return {}

def process_json_to_chunks(json_file_path: str, file_name: str, user_id: Optional[str] = None, file_description: Optional[str] = None, table_name: Optional[str] = None):
    """
    Main function to process JSON file and insert file and chunks into database.
    
    Args:
        json_file_path: Path to the JSON file
        file_name: Name of the file to use in metadata
        user_id: Optional user ID for user-specific database configuration
        file_description: Optional description for the file
        table_name: Optional table name for the file
    """
    
    # Database configuration - use user-specific config if provided
    db_config = get_database_config(user_id)
    
    # Create inserter and process
    inserter = ChunkDataInserter(db_config)
    
    try:
        logger.info(f"Starting to process JSON file: {json_file_path}")
        logger.info(f"File name for metadata: {file_name}")
        
        # Validate JSON fields first
        logger.info("🔍 Validating JSON field accessibility...")
        validation_result = inserter.validate_json_fields(json_file_path)
        
        # Insert file and chunks
        success = inserter.insert_file_and_chunks_from_json(json_file_path, file_name, file_description, table_name, user_id)
        
        if success:
            # Get the file_id from JSON to check results
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_id = data.get("file_id")
                
            if file_id:
                chunk_count = inserter.get_chunk_count(file_id)
                logger.info(f"✅ Processing completed successfully!")
                logger.info(f"📊 File ID: {file_id}")
                logger.info(f"📊 Total chunks inserted: {chunk_count}")
            else:
                logger.info("✅ Processing completed successfully!")
                
        else:
            logger.error("❌ Processing failed!")
            
    except Exception as e:
        logger.error(f"Error in process_json_to_chunks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys

    json_file_path='/Users/nilab/Desktop/projects/Knowladge-Base/processed_chunks_fixed.json'
    # Example usage with user_id (can be None for environment variable fallback)
    process_json_to_chunks(json_file_path, "file_name", user_id="nilab")