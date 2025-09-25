import os
import json
import logging
import time
import asyncio
import uuid
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
        if not validate_embedding(embedding1) or not validate_embedding(embedding2):
            return 0.0
        
        arr1 = np.array(embedding1, dtype=np.float32).reshape(1, -1)
        arr2 = np.array(embedding2, dtype=np.float32).reshape(1, -1)
        
        if arr1.shape[1] != arr2.shape[1]:
            logger.warning(f"Embedding dimension mismatch: {arr1.shape[1]} vs {arr2.shape[1]}")
            return 0.0
        
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        arr1_normalized = arr1 / norm1
        arr2_normalized = arr2 / norm2
        
        similarity = np.dot(arr1_normalized, arr2_normalized.T)[0][0]
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
        
        if isinstance(embedding_data, list):
            result = [float(x) for x in embedding_data]
        elif isinstance(embedding_data, np.ndarray):
            result = embedding_data.astype(np.float32).tolist()
        elif isinstance(embedding_data, str):
            try:
                parsed = json.loads(embedding_data)
                if isinstance(parsed, list):
                    result = [float(x) for x in parsed]
            except json.JSONDecodeError:
                pass
            
            if result is None:
                try:
                    clean_str = embedding_data.strip()
                    if clean_str.startswith('[') and clean_str.endswith(']'):
                        clean_str = clean_str[1:-1]
                        values = re.split(r'[,\s]+', clean_str)
                        result = [float(x) for x in values if x.strip()]
                except (ValueError, AttributeError):
                    pass
        
        if result is None:
            try:
                result = [float(x) for x in embedding_data]
            except (TypeError, ValueError, AttributeError):
                pass
        
        if result and validate_embedding(result):
            return result
        else:
            logger.warning(f"Invalid embedding data after conversion: length={len(result) if result else 0}")
            return None
        
    except Exception as e:
        logger.error(f"Error converting embedding: {e}")
        return None

@dataclass
class IntentMappingConfig:
    """Configuration for intent mapping processing."""
    api_key: str
    similarity_threshold: float = 0.75
    top_n_candidates: int = 10
    batch_size: int = 20
    delay_between_requests: float = 1.0
    max_retries: int = 3
    retry_delay: float = 2.0
    embedding_model: str = gemini_embedding_model_name
    output_dimensionality: int = 1536



@dataclass
class IntentMappingStats:
    """Statistics tracker for intent mapping processing."""
    chunks_total: int = 0
    chunks_processed: int = 0
    chunks_remaining: int = 0
    chunks_failed: int = 0
    
    intents_matched: int = 0
    intents_created: int = 0
    
    start_time: Optional[datetime] = None
    
    def get_completion_percentage(self) -> float:
        if self.chunks_total == 0:
            return 100.0
        return (self.chunks_processed / self.chunks_total) * 100

class IntentEmbeddingGenerator:
    """Handles embedding generation for intents."""
    
    def __init__(self, config: IntentMappingConfig):
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

class IntentDatabase:
    """Database operations for intent mapping."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._create_intent_schema()
    
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
    
    def _create_intent_schema(self):
        """Create intent tables and indexes if they don't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create intent_chunks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS intent_chunks (
                        intent_id VARCHAR(50) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL UNIQUE,
                        description TEXT NOT NULL,
                        embedding vector(1536),
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_intent_title ON intent_chunks (title)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_intent_embedding ON intent_chunks USING hnsw (embedding vector_cosine_ops)
                """)
                
                # Add columns to document_chunks table
                cursor.execute("""
                    ALTER TABLE document_chunks 
                    ADD COLUMN IF NOT EXISTS intent_id VARCHAR(50) REFERENCES intent_chunks(intent_id),
                    ADD COLUMN IF NOT EXISTS mapped_to_intent BOOLEAN DEFAULT FALSE
                """)
                
                conn.commit()
                logger.info("Intent schema created successfully")
                
        except Exception as e:
            logger.error(f"Error creating intent schema: {e}")
            raise
    
    def get_processing_status(self) -> Dict[str, int]:
        """Get current processing status."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total chunks
                cursor.execute("SELECT COUNT(*) FROM document_chunks")
                chunks_total = cursor.fetchone()[0]
                
                # Mapped chunks
                cursor.execute("SELECT COUNT(*) FROM document_chunks WHERE mapped_to_intent = TRUE")
                chunks_processed = cursor.fetchone()[0]
                
                # Unmapped chunks
                chunks_remaining = chunks_total - chunks_processed
                
                # Total intents
                cursor.execute("SELECT COUNT(*) FROM intent_chunks")
                intents_total = cursor.fetchone()[0]
                
                return {
                    "chunks_total": chunks_total,
                    "chunks_processed": chunks_processed,
                    "chunks_remaining": chunks_remaining,
                    "intents_total": intents_total
                }
                
        except Exception as e:
            logger.error(f"Error getting processing status: {e}")
            return {}
    
    def get_unmapped_chunks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get document chunks that haven't been mapped to intents."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT chunk_id, combined_context, embedding
                    FROM document_chunks 
                    WHERE mapped_to_intent = FALSE 
                    AND combined_context IS NOT NULL
                    AND embedding IS NOT NULL
                    ORDER BY chunk_id
                    LIMIT %s
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    row_dict = dict(row)
                    # Convert and validate embedding
                    embedding = convert_embedding_to_list(row_dict['embedding'])
                    
                    if embedding and validate_embedding(embedding):
                        row_dict['embedding'] = embedding
                        results.append(row_dict)
                    else:
                        logger.warning(f"Skipping chunk {row_dict['chunk_id']} due to invalid embedding")
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting unmapped chunks: {e}")
            return []
    
    def get_all_intents(self) -> List[Dict[str, Any]]:
        """Get all existing intents with embeddings."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute("""
                    SELECT intent_id, title, description, embedding
                    FROM intent_chunks
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
                        logger.warning(f"Intent {row_dict['intent_id']} has invalid embedding")
                        row_dict['embedding'] = None
                        results.append(row_dict)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting intents: {e}")
            return []
    
    def find_similar_intents(self, chunk_embedding: List[float], top_n: int = 5) -> List[Dict[str, Any]]:
        """Find top N similar intents using cosine similarity."""
        try:
            all_intents = self.get_all_intents()
            similarities = []
            
            for intent in all_intents:
                if intent['embedding']:
                    similarity = safe_cosine_similarity(chunk_embedding, intent['embedding'])
                    similarities.append({
                        'intent_id': intent['intent_id'],
                        'title': intent['title'],
                        'description': intent['description'],
                        'similarity': similarity
                    })
            
            # Sort by similarity and return top N
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_n]
            
        except Exception as e:
            logger.error(f"Error finding similar intents: {e}")
            return []
    
    def create_intent(self, title: str, description: str, embedding: List[float]) -> Optional[str]:
        """Create a new intent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                intent_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO intent_chunks (intent_id, title, description, embedding)
                    VALUES (%s, %s, %s, %s)
                    RETURNING intent_id
                """, (intent_id, title, description, embedding))
                
                created_intent_id = cursor.fetchone()[0]
                conn.commit()
                
                logger.info(f"Created new intent: {title} (ID: {created_intent_id})")
                return created_intent_id
                
        except Exception as e:
            logger.error(f"Error creating intent: {e}")
            return None
    
    def map_chunk_to_intent(self, chunk_id: str, intent_id: str) -> bool:
        """Map a chunk to an intent."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE document_chunks 
                    SET intent_id = %s, mapped_to_intent = TRUE
                    WHERE chunk_id = %s
                """, (intent_id, chunk_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error mapping chunk to intent: {e}")
            return False

class IntentClassifier:
    """Handles intent classification and creation."""
    
    def __init__(self, embedding_generator: IntentEmbeddingGenerator):
        self.embedding_generator = embedding_generator
    
    async def classify_chunk(self, chunk_context: str, similar_intents: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Classify a chunk using LLM with structured prompt."""
        
        # Build the prompt with similar intents
        intents_text = ""
        for i, intent in enumerate(similar_intents, 1):
            intents_text += f"{i}. Title: {intent['title']}\n   Description: {intent['description']}\n\n"
        
        prompt = f"""You are an AI sub-intent classifier.

Given:
- Document chunk:
\"\"\"
{chunk_context}
\"\"\"

And the following existing sub-intents:

{intents_text}

Decide:
- If this chunk fits any sub-intent, provide its title.
- If not, create a new sub-intent with a title and description.

Respond ONLY in the following JSON format:

{{
  "status": "fit" | "create",
  "title": "<if 'fit', existing title; if 'create', new title>",
  "description": "<if 'create', description; else null>"
}}"""
        
        try:
            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Clean the response to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Validate response structure
            if 'status' not in result or 'title' not in result:
                logger.error(f"Invalid response structure: {result}")
                return None
            
            if result['status'] not in ['fit', 'create']:
                logger.error(f"Invalid status: {result['status']}")
                return None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response was: {response_text}")
            return None
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return None

class IntentMappingManager:
    """Main manager for intent mapping processing."""
    
    def __init__(self, config: IntentMappingConfig, db_config: DatabaseConfig):
        self.config = config
        self.db_config = db_config
        self.embedding_generator = IntentEmbeddingGenerator(config)
        self.database = IntentDatabase(db_config)
        self.classifier = IntentClassifier(self.embedding_generator)
        self.stats = IntentMappingStats()
    
    def display_initial_status(self) -> bool:
        """Display initial processing status."""
        status = self.database.get_processing_status()
        
        self.stats.chunks_total = status.get("chunks_total", 0)
        self.stats.chunks_processed = status.get("chunks_processed", 0)
        self.stats.chunks_remaining = status.get("chunks_remaining", 0)
        
        print("\n" + "="*70)
        print("🎯 INTENT MAPPING PROCESSING STATUS")
        print("="*70)
        print(f"📄 CHUNKS:")
        print(f"   Total chunks in database: {self.stats.chunks_total}")
        print(f"   Chunks mapped to intents: {self.stats.chunks_processed}")
        print(f"   Chunks WITHOUT intent mapping: {self.stats.chunks_remaining}")
        print()
        print(f"🏷️  INTENTS:")
        print(f"   Total intents available: {status.get('intents_total', 0)}")
        print("="*70)
        
        if self.stats.chunks_remaining == 0:
            print("✅ All chunks already have intent mappings!")
            return False
        else:
            print(f"🚀 Ready to process {self.stats.chunks_remaining} chunks")
            return True
    
    def display_progress(self):
        """Display current progress."""
        if self.stats.chunks_total > 0:
            progress_pct = self.stats.get_completion_percentage()
            print(f"📈 PROGRESS: {self.stats.chunks_processed}/{self.stats.chunks_total} ({progress_pct:.1f}%) | "
                  f"Remaining: {self.stats.chunks_remaining} | Failed: {self.stats.chunks_failed} | "
                  f"Matched: {self.stats.intents_matched} | Created: {self.stats.intents_created}")
    
    async def process_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """Process a single chunk for intent mapping."""
        chunk_id = chunk_data['chunk_id']
        chunk_context = chunk_data['combined_context']
        chunk_embedding = chunk_data['embedding']
        
        logger.info(f"Processing chunk: {chunk_id}")
        
        # Find similar intents
        similar_intents = self.database.find_similar_intents(
            chunk_embedding, 
            top_n=self.config.top_n_candidates
        )
        
        # Only proceed if we have similar intents above threshold
        filtered_intents = [
            intent for intent in similar_intents 
            if intent['similarity'] >= self.config.similarity_threshold
        ]
        
        # Use all intents if none meet threshold (for LLM to decide)
        intents_for_llm = filtered_intents if filtered_intents else similar_intents
        
        # Classify using LLM
        classification = await self.classifier.classify_chunk(chunk_context, intents_for_llm)
        
        if not classification:
            logger.error(f"Failed to classify chunk: {chunk_id}")
            return False
        
        if classification['status'] == 'fit':
            # Find the intent ID for the matched title
            intent_id = None
            for intent in intents_for_llm:
                if intent['title'] == classification['title']:
                    intent_id = intent['intent_id']
                    break
            
            if not intent_id:
                logger.error(f"Could not find intent ID for title: {classification['title']}")
                return False
            
            # Map chunk to existing intent
            success = self.database.map_chunk_to_intent(chunk_id, intent_id)
            if success:
                self.stats.intents_matched += 1
                logger.info(f"Mapped chunk {chunk_id} to existing intent: {classification['title']}")
                return True
            else:
                logger.error(f"Failed to map chunk to intent")
                return False
        
        elif classification['status'] == 'create':
            # Create new intent
            if not classification.get('description'):
                logger.error(f"Missing description for new intent: {classification['title']}")
                return False
            
            # Generate embedding for new intent
            intent_text = f"Title: {classification['title']}\nDescription: {classification['description']}"
            embedding = await self.embedding_generator.generate_embedding(intent_text)
            
            if not embedding:
                logger.error(f"Failed to generate embedding for new intent")
                return False
            
            # Create intent in database
            intent_id = self.database.create_intent(
                classification['title'],
                classification['description'],
                embedding
            )
            
            if not intent_id:
                logger.error(f"Failed to create new intent")
                return False
            
            # Map chunk to new intent
            success = self.database.map_chunk_to_intent(chunk_id, intent_id)
            if success:
                self.stats.intents_created += 1
                logger.info(f"Created new intent '{classification['title']}' and mapped chunk {chunk_id}")
                return True
            else:
                logger.error(f"Failed to map chunk to new intent")
                return False
        
        else:
            logger.error(f"Unknown classification status: {classification['status']}")
            return False
    
    async def process_all_chunks(self):
        """Process all unmapped chunks."""
        needs_processing = self.display_initial_status()
        
        if not needs_processing:
            return
        
        self.stats.start_time = datetime.now()
        
        # Process chunks in batches
        total_processed = 0
        
        while self.stats.chunks_remaining > 0:
            # Get batch of unmapped chunks
            chunks_to_process = self.database.get_unmapped_chunks(limit=self.config.batch_size)
            
            if not chunks_to_process:
                print("No more chunks to process")
                break
            
            print(f"\n🔄 Processing batch of {len(chunks_to_process)} chunks...")
            
            for chunk_data in chunks_to_process:
                try:
                    success = await self.process_chunk(chunk_data)
                    
                    if success:
                        self.stats.chunks_processed += 1
                        total_processed += 1
                    else:
                        self.stats.chunks_failed += 1
                    
                    # Update remaining count
                    self.stats.chunks_remaining = self.stats.chunks_total - self.stats.chunks_processed - self.stats.chunks_failed
                    
                    # Display progress every 10 chunks
                    if total_processed % 10 == 0:
                        self.display_progress()
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_data.get('chunk_id', 'unknown')}: {e}")
                    self.stats.chunks_failed += 1
                    self.stats.chunks_remaining = self.stats.chunks_total - self.stats.chunks_processed - self.stats.chunks_failed
            
            # Small delay between batches
            await asyncio.sleep(1)
        
        # Final summary
        self.display_final_summary()
    
    def display_final_summary(self):
        """Display final processing summary."""
        elapsed_time = datetime.now() - self.stats.start_time if self.stats.start_time else None
        elapsed_str = f"{elapsed_time.total_seconds():.1f}s" if elapsed_time else "N/A"
        
        print("\n" + "="*70)
        print("🎯 INTENT MAPPING COMPLETE!")
        print("="*70)
        print(f"⏱️  Total time: {elapsed_str}")
        print(f"📊 Chunks processed: {self.stats.chunks_processed}")
        print(f"❌ Chunks failed: {self.stats.chunks_failed}")
        print(f"📋 Chunks remaining: {self.stats.chunks_remaining}")
        print()
        print(f"🔗 Chunks matched to existing intents: {self.stats.intents_matched}")
        print(f"🆕 New intents created: {self.stats.intents_created}")
        print()
        
        if self.stats.chunks_failed > 0:
            print(f"⚠️  {self.stats.chunks_failed} chunks failed processing - check logs for details")
        
        if self.stats.chunks_remaining == 0:
            print("✅ All chunks now have intent mappings!")
        else:
            print(f"📝 {self.stats.chunks_remaining} chunks still need processing")
        
        print("="*70)

async def intent_mapping_main(user_id: Optional[str] = None):
    """Main execution function."""
    print("🎯 Starting Intent Mapping System")
    print("="*50)
    
    # Configuration
    intent_config = IntentMappingConfig(
        api_key=gemini_apikey,
        similarity_threshold=0.70,
        top_n_candidates=5,
        batch_size=20,
        delay_between_requests=1.0,
        max_retries=3,
        retry_delay=2.0
    )
    
    # Load database config from API or environment variables
    db_config = get_database_config(user_id)
    
    # Initialize manager
    manager = IntentMappingManager(intent_config, db_config)
    
    try:
        # Process all chunks
        await manager.process_all_chunks()
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        manager.display_progress()
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {e}")
        print(f"\n❌ Fatal error: {e}")
        
    finally:
        print("\n👋 Intent Mapping System shutting down...")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(intent_mapping_main())