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
from sklearn.metrics.pairwise import cosine_similarity

# LLM Setup
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the new config loader
from data_sources.file_data.app.unstructured.agent.config_loader import get_database_config, DatabaseConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize LLM
load_dotenv(override=True)
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
google_gemini_name_light = os.getenv("google_gemini_name_light", "gemini-1.5-pro")
thinking_model = os.getenv("thinking_model", "deepseek-r1-distill-llama-70b")

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """Initialize and return the ChatGoogleGenerativeAI LLM instance."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

llm = initialize_llm_gemini()
gemini_embedding_model_name = os.getenv("google_gemini_embedding_name", "gemini-embedding-exp-03-07")
if gemini_embedding_model_name is None:
    print("Warning: 'google_gemini_embedding_name' not found in environment variables, using default 'gemini-embedding-exp-03-07'.")
# Setup logging

def validate_embedding(embedding: List[float]) -> bool:
    """Validate that embedding is suitable for similarity calculations."""
    if not embedding or len(embedding) == 0:
        return False
    
    # Convert to numpy array for validation
    arr = np.array(embedding, dtype=np.float32)
    
    # Check for NaN or infinite values
    if np.isnan(arr).any() or np.isinf(arr).any():
        logger.warning("Embedding contains NaN or infinite values")
        return False
    
    # Check if it's a zero vector
    if np.allclose(arr, 0):
        logger.warning("Embedding is a zero vector")
        return False
    
    # Check for extreme values that might cause overflow
    if np.abs(arr).max() > 1e6:
        logger.warning("Embedding contains extremely large values")
        return False
    
    return True

def safe_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity with proper error handling."""
    try:
        # Validate both embeddings
        if not validate_embedding(embedding1) or not validate_embedding(embedding2):
            return 0.0
        
        # Convert to numpy arrays with proper dtype
        arr1 = np.array(embedding1, dtype=np.float32).reshape(1, -1)
        arr2 = np.array(embedding2, dtype=np.float32).reshape(1, -1)
        
        # Check dimensions match
        if arr1.shape[1] != arr2.shape[1]:
            logger.warning(f"Embedding dimension mismatch: {arr1.shape[1]} vs {arr2.shape[1]}")
            return 0.0
        
        # Normalize vectors to prevent overflow
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        arr1_normalized = arr1 / norm1
        arr2_normalized = arr2 / norm2
        
        # Calculate cosine similarity manually to avoid sklearn warnings
        similarity = np.dot(arr1_normalized, arr2_normalized.T)[0][0]
        
        # Clip to valid range [-1, 1] to handle floating point errors
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def convert_embedding_to_list(embedding_data) -> Optional[List[float]]:
    """Convert various embedding formats to list of floats with validation."""
    if embedding_data is None:
        return None
    
    try:
        result = None
        
        # If it's already a list, validate and return
        if isinstance(embedding_data, list):
            result = [float(x) for x in embedding_data]
        
        # If it's a numpy array
        elif isinstance(embedding_data, np.ndarray):
            result = embedding_data.astype(np.float32).tolist()
        
        # If it's a string representation of a list or array
        elif isinstance(embedding_data, str):
            # Try to parse as JSON array
            try:
                parsed = json.loads(embedding_data)
                if isinstance(parsed, list):
                    result = [float(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            
            if result is None:
                # Try to parse as numpy array string representation
                try:
                    # Remove any numpy array formatting
                    clean_str = embedding_data.strip()
                    if clean_str.startswith('[') and clean_str.endswith(']'):
                        # Remove brackets and split by whitespace/comma
                        clean_str = clean_str[1:-1]
                        values = re.split(r'[,\s]+', clean_str)
                        result = [float(x) for x in values if x.strip()]
                except (ValueError, AttributeError):
                    pass
        
        # If it's some other iterable, try to convert
        if result is None:
            try:
                result = [float(x) for x in embedding_data]
            except (TypeError, ValueError, AttributeError):
                pass
        
        # Validate the result
        if result and validate_embedding(result):
            return result
        else:
            logger.warning(f"Invalid embedding data after conversion: length={len(result) if result else 0}")
            return None
        
    except Exception as e:
        logger.error(f"Error converting embedding: {e}")
        return None

@dataclass
class SubIntentConfig:
    """Configuration for sub-intent processing."""
    api_key: str
    similarity_threshold: float = 0.75  # Threshold for matching existing sub-intents
    batch_size: int = 10
    delay_between_requests: float = 1.5
    max_retries: int = 3
    retry_delay: float = 2.0
    embedding_model: str = gemini_embedding_model_name
    output_dimensionality: int = 1536




@dataclass
class ProcessingStats:
    """Processing statistics tracker."""
    files_total: int = 0
    files_processed: int = 0
    files_remaining: int = 0
    files_failed: int = 0
    
    sub_intents_created: int = 0
    sub_intents_matched: int = 0
    
    start_time: Optional[datetime] = None
    
    def get_completion_percentage(self) -> float:
        if self.files_total == 0:
            return 100.0
        return (self.files_processed / self.files_total) * 100

class EmbeddingGenerator:
    """Handles embedding generation for sub-intents."""
    
    def __init__(self, config: SubIntentConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text with validation."""
        for attempt in range(self.config.max_retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_delay * attempt)
                
                result = genai.embed_content(
                    model=self.config.embedding_model,
                    content=text,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.config.output_dimensionality
                )
                
                await asyncio.sleep(self.config.delay_between_requests)
                
                # Validate the embedding before returning
                embedding = result.get('embedding', [])
                if validate_embedding(embedding):
                    return embedding
                else:
                    logger.warning(f"Generated embedding failed validation on attempt {attempt + 1}")
                    continue
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts: {e}")
                    return None
        
        return None

class SubIntentDatabase:
    """Database operations for sub-intents."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._create_sub_intent_table()
    
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
    
    def _create_sub_intent_table(self):
        """Create sub_intents table if it doesn't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sub_intents (
                        sub_intent_id SERIAL PRIMARY KEY,
                        title VARCHAR(255) NOT NULL UNIQUE,
                        description TEXT NOT NULL,
                        file_ids TEXT[] DEFAULT '{}',
                        embedding vector(1536) DEFAULT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                                
                # Add sub_intent_id column to document_files if it doesn't exist
                cursor.execute("""
                    ALTER TABLE document_files 
                    ADD COLUMN IF NOT EXISTS sub_intent_id INTEGER REFERENCES sub_intents(sub_intent_id)
                """)
                
                conn.commit()
                logger.info("Sub-intents table and file reference created successfully")
                
        except Exception as e:
            logger.error(f"Error creating sub_intents table: {e}")
            raise
    
    def get_processing_status(self) -> Dict[str, int]:
        """Get current processing status."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total files
                cursor.execute("SELECT COUNT(*) FROM document_files")
                files_total = cursor.fetchone()[0]
                
                # Files with sub-intent assigned
                cursor.execute("SELECT COUNT(*) FROM document_files WHERE sub_intent_id IS NOT NULL")
                files_processed = cursor.fetchone()[0]
                
                # Files without sub-intent
                files_remaining = files_total - files_processed
                
                # Total sub-intents
                cursor.execute("SELECT COUNT(*) FROM sub_intents")
                sub_intents_total = cursor.fetchone()[0]
                
                return {
                    "files_total": files_total,
                    "files_processed": files_processed,
                    "files_remaining": files_remaining,
                    "sub_intents_total": sub_intents_total
                }
                
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {}
    
    def get_files_without_sub_intent(self) -> List[Dict[str, Any]]:
        """Get files that don't have sub-intent assigned and have valid embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT file_id, title, full_summary, title_summary_embedding
                    FROM document_files 
                    WHERE sub_intent_id IS NULL 
                    AND title_summary_embedding IS NOT NULL
                    ORDER BY processing_timestamp DESC
                """)
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Convert and validate embedding
                    embedding = convert_embedding_to_list(row_dict['title_summary_embedding'])
                    
                    if embedding and validate_embedding(embedding):
                        row_dict['title_summary_embedding'] = embedding
                        results.append(row_dict)
                    else:
                        logger.warning(f"Skipping file {row_dict['file_id']} due to invalid embedding")
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting files without sub-intent: {e}")
            return []
    
    def get_all_sub_intents(self) -> List[Dict[str, Any]]:
        """Get all existing sub-intents with valid embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT sub_intent_id, title, description, file_ids, embedding
                    FROM sub_intents
                    ORDER BY created_at
                """)
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Convert and validate embedding
                    embedding = convert_embedding_to_list(row_dict['embedding'])
                    
                    if embedding and validate_embedding(embedding):
                        row_dict['embedding'] = embedding
                        results.append(row_dict)
                    else:
                        logger.warning(f"Sub-intent {row_dict['sub_intent_id']} has invalid embedding, skipping from similarity calculations")
                        # Still include it but with None embedding so it won't be used for similarity
                        row_dict['embedding'] = None
                        results.append(row_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting sub-intents: {e}")
            return []
    
    def create_sub_intent(self, title: str, description: str, embedding: List[float], file_id: str) -> Optional[int]:
        """Create a new sub-intent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO sub_intents (title, description, file_ids, embedding)
                    VALUES (%s, %s, %s, %s)
                    RETURNING sub_intent_id
                """, (title, description, [file_id], embedding))
                
                sub_intent_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created new sub-intent: {title} (ID: {sub_intent_id})")
                return sub_intent_id
                
        except Exception as e:
            logger.error(f"Error creating sub-intent: {e}")
            return None
    
    def add_file_to_sub_intent(self, sub_intent_id: int, file_id: str) -> bool:
        """Add file to existing sub-intent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Add file_id to the file_ids array
                cursor.execute("""
                    UPDATE sub_intents 
                    SET file_ids = array_append(file_ids, %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE sub_intent_id = %s
                """, (file_id, sub_intent_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error adding file to sub-intent: {e}")
            return False
    
    def assign_file_to_sub_intent(self, file_id: str, sub_intent_id: int) -> bool:
        """Assign file to sub-intent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE document_files 
                    SET sub_intent_id = %s
                    WHERE file_id = %s
                """, (sub_intent_id, file_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error assigning file to sub-intent: {e}")
            return False

class SubIntentClassifier:
    """Handles classification and creation of sub-intents."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    async def find_matching_sub_intent(self, file_title: str, file_summary: str, 
                                     file_embedding: List[float],
                                     existing_sub_intents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find best matching sub-intent using file embedding and LLM confirmation."""
        
        if not existing_sub_intents or not file_embedding:
            return None
        
        # Calculate similarities with existing sub-intents
        best_match = None
        best_similarity = 0.0
        
        for sub_intent in existing_sub_intents:
            if sub_intent['embedding'] and len(sub_intent['embedding']) > 0:
                try:
                    # Use the safe cosine similarity function
                    similarity = safe_cosine_similarity(file_embedding, sub_intent['embedding'])
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = sub_intent
                        
                except Exception as e:
                    logger.error(f"Error calculating similarity for sub-intent {sub_intent.get('sub_intent_id', 'unknown')}: {e}")
                    continue
        
        # If similarity is above threshold, use LLM to confirm
        if best_match and best_similarity > 0.65:  # Lower threshold for LLM confirmation
            confirmation_prompt = f"""
            Analyze if this document should be classified under the existing sub-intent:

            DOCUMENT:
            Title: {file_title}
            Summary: {file_summary}

            EXISTING SUB-INTENT:
            Title: {best_match['title']}
            Description: {best_match['description']}

            Question: Does this document belong to this sub-intent category?
            
            Respond with only "YES" or "NO" followed by a brief reason.
            """
            
            try:
                response = llm.invoke(confirmation_prompt)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                if response_text and response_text.strip().upper().startswith('YES'):
                    logger.info(f"LLM confirmed match: {file_title} -> {best_match['title']} (similarity: {best_similarity:.3f})")
                    return best_match
                else:
                    logger.info(f"LLM rejected match: {file_title} -> {best_match['title']} (similarity: {best_similarity:.3f})")
            except Exception as e:
                logger.error(f"Error in LLM confirmation: {e}")
        
        return None
    
    async def generate_new_sub_intent(self, file_title: str, file_summary: str) -> Optional[Dict[str, str]]:
        """Generate a new sub-intent for the file."""
        
        prompt = f"""
        Based on the following document, create a new sub-intent category:

        DOCUMENT:
        Title: {file_title}
        Summary: {file_summary}

        Create a sub-intent that:
        1. Captures the main theme/purpose of this document
        2. Is broad enough to include similar documents
        3. Is specific enough to be meaningful

        Provide your response in this exact format:
        TITLE: [Short, descriptive title for the sub-intent]
        DESCRIPTION: [Detailed description of what types of documents belong to this category]

        Example:
        TITLE: Financial Planning Documents
        DESCRIPTION: Documents related to financial planning, budgeting, investment strategies, and economic analysis for personal or business purposes.
        """
        
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            if not response_text:
                return None
            
            lines = response_text.strip().split('\n')
            title = None
            description = None
            
            for line in lines:
                if line.startswith('TITLE:'):
                    title = line.replace('TITLE:', '').strip()
                elif line.startswith('DESCRIPTION:'):
                    description = line.replace('DESCRIPTION:', '').strip()
            
            if title and description:
                return {"title": title, "description": description}
            else:
                logger.warning(f"Failed to parse LLM response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating sub-intent: {e}")
            return None

class SubIntentManager:
    """Main manager for sub-intent processing."""
    
    def __init__(self, config: SubIntentConfig, db_config: DatabaseConfig):
        self.config = config
        self.db_config = db_config
        self.embedding_generator = EmbeddingGenerator(config)
        self.database = SubIntentDatabase(db_config)
        self.classifier = SubIntentClassifier(self.embedding_generator)
        self.stats = ProcessingStats()
    
    def display_initial_status(self) -> bool:
        """Display initial processing status."""
        status = self.database.get_processing_status()
        
        self.stats.files_total = status.get("files_total", 0)
        self.stats.files_processed = status.get("files_processed", 0)
        self.stats.files_remaining = status.get("files_remaining", 0)
        
        print("\n" + "="*70)
        print("📊 SUB-INTENT PROCESSING STATUS")
        print("="*70)
        print(f"📄 FILES:")
        print(f"   Total files in database: {self.stats.files_total}")
        print(f"   Files with sub-intent assigned: {self.stats.files_processed}")
        print(f"   Files WITHOUT sub-intent: {self.stats.files_remaining}")
        print()
        print(f"🏷️  SUB-INTENTS:")
        print(f"   Total sub-intents created: {status.get('sub_intents_total', 0)}")
        print("="*70)
        
        if self.stats.files_remaining == 0:
            print("✅ All files already have sub-intents assigned!")
            return False
        else:
            print(f"🚀 Ready to process {self.stats.files_remaining} files")
            return True
    
    def display_progress(self):
        """Display current progress."""
        if self.stats.files_total > 0:
            progress_pct = self.stats.get_completion_percentage()
            print(f"📈 PROGRESS: {self.stats.files_processed}/{self.stats.files_total} ({progress_pct:.1f}%) | "
                  f"Remaining: {self.stats.files_remaining} | Failed: {self.stats.files_failed} | "
                  f"New Sub-intents: {self.stats.sub_intents_created} | Matched: {self.stats.sub_intents_matched}")
    
    async def process_file(self, file_data: Dict[str, Any], existing_sub_intents: List[Dict[str, Any]]) -> bool:
        """Process a single file for sub-intent assignment."""
        file_id = file_data['file_id']
        file_title = file_data['title']
        file_summary = file_data['full_summary']
        file_embedding = file_data['title_summary_embedding']
        
        logger.info(f"Processing file: {file_title}")
        
        # Validate file embedding
        if not file_embedding or not validate_embedding(file_embedding):
            logger.error(f"File {file_id} has no valid embedding")
            return False
        
        # Try to find matching sub-intent using existing file embedding
        matching_sub_intent = await self.classifier.find_matching_sub_intent(
            file_title, file_summary, file_embedding, existing_sub_intents
        )
        
        if matching_sub_intent:
            # Assign to existing sub-intent
            sub_intent_id = matching_sub_intent['sub_intent_id']
            
            success = (
                self.database.add_file_to_sub_intent(sub_intent_id, file_id) and
                self.database.assign_file_to_sub_intent(file_id, sub_intent_id)
            )
            
            if success:
                self.stats.sub_intents_matched += 1
                logger.info(f"File '{file_title}' assigned to existing sub-intent: {matching_sub_intent['title']}")
                return True
            else:
                logger.error(f"Failed to assign file to existing sub-intent")
                return False
        
        else:
            # Generate new sub-intent
            new_sub_intent = await self.classifier.generate_new_sub_intent(file_title, file_summary)
            
            if not new_sub_intent:
                logger.error(f"Failed to generate new sub-intent for file: {file_title}")
                return False
            
            # Create embedding for new sub-intent
            sub_intent_text = f"Title: {new_sub_intent['title']}\nDescription: {new_sub_intent['description']}"
            embedding = await self.embedding_generator.generate_embedding(sub_intent_text)
            
            if not embedding or not validate_embedding(embedding):
                logger.error(f"Failed to generate valid embedding for new sub-intent")
                return False
            
            # Create new sub-intent in database
            sub_intent_id = self.database.create_sub_intent(
                new_sub_intent['title'],
                new_sub_intent['description'],
                embedding,
                file_id
            )
            
            if sub_intent_id:
                # Assign file to new sub-intent
                success = self.database.assign_file_to_sub_intent(file_id, sub_intent_id)
                
                if success:
                    self.stats.sub_intents_created += 1
                    # Add to existing sub-intents for next iterations
                    existing_sub_intents.append({
                        'sub_intent_id': sub_intent_id,
                        'title': new_sub_intent['title'],
                        'description': new_sub_intent['description'],
                        'embedding': embedding,
                        'file_ids': [file_id]
                    })
                    
                    logger.info(f"Created new sub-intent '{new_sub_intent['title']}' for file: {file_title}")
                    return True
                else:
                    logger.error(f"Failed to assign file to new sub-intent")
                    return False
            else:
                logger.error(f"Failed to create new sub-intent")
                return False
    
    async def process_all_files(self):
        """Process all files without sub-intents."""
        needs_processing = self.display_initial_status()
        
        if not needs_processing:
            return
        
        self.stats.start_time = datetime.now()
        
        # Get files to process (only those with valid embeddings)
        files_to_process = self.database.get_files_without_sub_intent()
        
        if not files_to_process:
            print("No files found to process (files need to have valid title_summary_embedding)")
            return
        
        # Get existing sub-intents
        existing_sub_intents = self.database.get_all_sub_intents()
        
        print(f"\n🔄 Starting processing of {len(files_to_process)} files...")
        print(f"📋 Found {len(existing_sub_intents)} existing sub-intents")
        
        # Process files in batches
        for i in range(0, len(files_to_process), self.config.batch_size):
            batch = files_to_process[i:i + self.config.batch_size]
            
            for file_data in batch:
                try:
                    success = await self.process_file(file_data, existing_sub_intents)
                    
                    if success:
                        self.stats.files_processed += 1
                    else:
                        self.stats.files_failed += 1
                    
                    # Update remaining count
                    self.stats.files_remaining = self.stats.files_total - self.stats.files_processed - self.stats.files_failed                    
                except Exception as e:
                    logger.error(f"Error processing file {file_data.get('file_id', 'unknown')}: {e}")
                    self.stats.files_failed += 1
                    # Update remaining count
                    self.stats.files_remaining = self.stats.files_total - self.stats.files_processed - self.stats.files_failed 
            
            # Display progress after each batch
            self.display_progress()
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Final summary
        self.display_final_summary()
    
    def display_final_summary(self):
        """Display final processing summary."""
        elapsed_time = datetime.now() - self.stats.start_time if self.stats.start_time else None
        elapsed_str = f"{elapsed_time.total_seconds():.1f}s" if elapsed_time else "N/A"
        
        print("\n" + "="*70)
        print("🎯 PROCESSING COMPLETE!")
        print("="*70)
        print(f"⏱️  Total time: {elapsed_str}")
        print(f"📊 Files processed: {self.stats.files_processed}")
        print(f"❌ Files failed: {self.stats.files_failed}")
        print(f"📋 Files remaining: {self.stats.files_remaining}")
        print()
        print(f"🆕 New sub-intents created: {self.stats.sub_intents_created}")
        print(f"🔗 Files matched to existing sub-intents: {self.stats.sub_intents_matched}")
        print()
        
        if self.stats.files_failed > 0:
            print(f"⚠️  {self.stats.files_failed} files failed processing - check logs for details")
        
        if self.stats.files_remaining == 0:
            print("✅ All files now have sub-intents assigned!")
        else:
            print(f"📝 {self.stats.files_remaining} files still need processing")
        
        print("="*70)

async def sub_intent_gen(user_id: Optional[str] = None):
    """Main execution function."""
    print("🚀 Starting Sub-Intent Processing System")
    print("="*50)
    
    # Configuration
    sub_intent_config = SubIntentConfig(
        api_key=gemini_apikey,
        similarity_threshold=0.75,
        batch_size=5,  # Process 5 files at a time
        delay_between_requests=1.5,
        max_retries=3,
        retry_delay=2.0
    )
    
    # Load database config from API or environment variables
    db_config = get_database_config(user_id)

    
    # Initialize manager
    manager = SubIntentManager(sub_intent_config, db_config)
    
    try:
        # Process all files
        await manager.process_all_files()
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        manager.display_progress()
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        print("\n👋 Sub-Intent Processing System shutting down...")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(sub_intent_gen())