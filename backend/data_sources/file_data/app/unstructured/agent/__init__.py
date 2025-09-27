import asyncio
import asyncpg
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5433
    database: str = "document_db"
    username: str = "postgres"
    password: str = "1234"

    def get_connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class EmbeddingGenerator:
    """Generates embeddings using Google Generative AI embeddings API."""

    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-exp-03-07", output_dim: int = 1536):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.output_dim = output_dim

    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY", retries: int = 3, delay: float = 1.5) -> Optional[List[float]]:
        """Generate embedding for a given text."""
        for attempt in range(retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(delay * attempt)

                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type=task_type,
                    output_dimensionality=self.output_dim
                )

                embedding = result.get('embedding', [])
                if self.validate_embedding(embedding):
                    return embedding
                else:
                    logger.warning(f"Generated embedding failed validation on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Embedding attempt {attempt + 1} failed: {e}")

        logger.error(f"Failed to generate embedding after {retries} attempts for text: {text[:30]}...")
        return None

    @staticmethod
    def validate_embedding(embedding: List[float]) -> bool:
        """Check if embedding is a valid non-zero, finite vector."""
        if not embedding or len(embedding) == 0:
            return False
        arr = np.array(embedding, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any() or np.allclose(arr, 0) or np.abs(arr).max() > 1e6:
            return False
        return True

class VectorSearcher:
    """Performs vector search on sub_intents table."""
    
    def __init__(self, db_config: DatabaseConfig, embedding_generator: EmbeddingGenerator):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
    
    async def search_similar_intents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar sub-intents using vector similarity."""
        
        # Generate embedding for query
        query_embedding = await self.embedding_generator.generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            return []
        
        # Connect to database and perform vector search
        conn = await asyncpg.connect(self.db_config.get_connection_string())
        
        try:
            # Convert embedding to string format for PostgreSQL
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Perform cosine similarity search using pgvector
            query_sql = """
                SELECT 
                    sub_intent_id,
                    title,
                    description,
                    file_ids,
                    1 - (embedding <=> %s::vector) as similarity
                FROM sub_intents
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            results = await conn.fetch(query_sql, embedding_str, embedding_str, top_k)
            
            # Convert results to list of dictionaries
            search_results = []
            for row in results:
                search_results.append({
                    'sub_intent_id': row['sub_intent_id'],
                    'title': row['title'],
                    'description': row['description'],
                    'file_ids': row['file_ids'],
                    'similarity': float(row['similarity'])
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return []
        finally:
            await conn.close()

async def main():
    """Main function to demonstrate vector search."""
    
    # Configuration
    GOOGLE_API_KEY = "your-google-api-key-here"  # Replace with your actual API key
    db_config = DatabaseConfig()
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(GOOGLE_API_KEY)
    vector_searcher = VectorSearcher(db_config, embedding_generator)
    
    # Example query
    user_query = "How do I get a driver's license?"
    
    print(f"Searching for: {user_query}")
    
    # Perform vector search
    results = await vector_searcher.search_similar_intents(user_query, top_k=3)
    
    # Display results
    if results:
        print(f"\nFound {len(results)} similar intents:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['title']} (ID: {result['sub_intent_id']})")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Description: {result['description'][:100]}...")
            print(f"   File IDs: {result['file_ids']}")
    else:
        print("No similar intents found.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())