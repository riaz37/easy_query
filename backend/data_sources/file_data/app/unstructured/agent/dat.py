import os
import asyncio
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import google.generativeai as genai
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======= Database Config =======
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

# ======= Embedding Generator =======
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

# ======= Vector Search Functions =======
class VectorSearchEngine:
    """Handles all vector search operations across different tables."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def search_similar_subintents(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar sub-intents based on query embedding."""
        sql = """
            SELECT 
                sub_intent_id, 
                title, 
                description, 
                file_ids,
                embedding <=> %s::vector AS similarity_score
            FROM sub_intents
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (query_embedding, query_embedding, top_k))
                    results = cursor.fetchall()
                    logger.info(f"Found {len(results)} similar sub-intents")
                    return results
        except Exception as e:
            logger.error(f"Database error in sub-intents search: {e}")
            return []

    def search_similar_documents(self, query_embedding: List[float], sub_intent_ids: List[int], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar documents based on sub-intent IDs and query embedding."""
        if not sub_intent_ids:
            return []
            
        sql = """
            SELECT 
                file_id,
                file_name,
                file_type,
                title,
                full_summary,
                intent,
                sub_intent,
                sub_intent_id,
                title_summary_embedding <=> %s::vector AS similarity_score
            FROM document_files
            WHERE sub_intent_id = ANY(%s)
            ORDER BY title_summary_embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (query_embedding, sub_intent_ids, query_embedding, top_k))
                    results = cursor.fetchall()
                    logger.info(f"Found {len(results)} similar documents")
                    return results
        except Exception as e:
            logger.error(f"Database error in documents search: {e}")
            return []

    def search_similar_chunks_by_embedding(self, query_embedding: List[float], file_ids: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for similar chunks using the main embedding field."""
        if not file_ids:
            return []
            
        sql = """
            SELECT 
                chunk_id,
                file_id,
                chunk_text,
                summary,
                title,
                keywords,
                chunk_order,
                combined_context,
                embedding <=> %s::vector AS similarity_score
            FROM document_chunks
            WHERE file_id = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (query_embedding, file_ids, query_embedding, top_k))
                    results = cursor.fetchall()
                    logger.info(f"Found {len(results)} similar chunks using embedding field")
                    return results
        except Exception as e:
            logger.error(f"Database error in chunks search (embedding): {e}")
            return []

    def search_similar_chunks_by_combined_embedding(self, query_embedding: List[float], file_ids: List[str], top_k: int = 20) -> List[Dict[str, Any]]:
        """Search for similar chunks using the combined_embedding field."""
        if not file_ids:
            return []
            
        sql = """
            SELECT 
                chunk_id,
                file_id,
                chunk_text,
                summary,
                title,
                keywords,
                chunk_order,
                combined_context,
                combined_embedding <=> %s::vector AS similarity_score
            FROM document_chunks
            WHERE file_id = ANY(%s)
            ORDER BY combined_embedding <=> %s::vector
            LIMIT %s;
        """
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (query_embedding, file_ids, query_embedding, top_k))
                    results = cursor.fetchall()
                    logger.info(f"Found {len(results)} similar chunks using combined_embedding field")
                    return results
        except Exception as e:
            logger.error(f"Database error in chunks search (combined_embedding): {e}")
            return []

# ======= Main Search Pipeline =======
class DocumentSearchPipeline:
    """Complete pipeline for multi-stage document search."""
    
    def __init__(self, db_config: DatabaseConfig, embedding_generator: EmbeddingGenerator):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.search_engine = VectorSearchEngine(db_config)
    
    async def search(self, query: str, 
                    subintent_top_k: int = 3, 
                    document_top_k: int = 5, 
                    chunk_top_k: int = 10) -> Dict[str, Any]:
        """
        Complete search pipeline:
        1. Generate embedding for query
        2. Search sub-intents
        3. Search documents from matching sub-intents
        4. Search chunks from matching documents (both embedding types)
        """
        
        logger.info(f"Starting search pipeline for query: {query[:50]}...")
        
        # Step 1: Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # Step 2: Search sub-intents
        logger.info("Step 1: Searching sub-intents...")
        subintent_results = self.search_engine.search_similar_subintents(query_embedding, subintent_top_k)
        if not subintent_results:
            return {"error": "No matching sub-intents found"}
        
        # Extract sub-intent IDs
        sub_intent_ids = [result['sub_intent_id'] for result in subintent_results]
        
        # Step 3: Search documents
        logger.info("Step 2: Searching documents...")
        document_results = self.search_engine.search_similar_documents(query_embedding, sub_intent_ids, document_top_k)
        if not document_results:
            return {
                "query": query,
                "subintent_results": subintent_results,
                "document_results": [],
                "chunk_results_embedding": [],
                "chunk_results_combined": [],
                "message": "No matching documents found"
            }
        
        # Extract file IDs
        file_ids = [result['file_id'] for result in document_results]
        
        # Step 4: Search chunks (both embedding types)
        logger.info("Step 3: Searching chunks...")
        chunk_results_embedding = self.search_engine.search_similar_chunks_by_embedding(query_embedding, file_ids, chunk_top_k)
        chunk_results_combined = self.search_engine.search_similar_chunks_by_combined_embedding(query_embedding, file_ids, chunk_top_k)
        
        return {
            "query": query,
            "subintent_results": subintent_results,
            "document_results": document_results,
            "chunk_results_embedding": chunk_results_embedding,
            "chunk_results_combined": chunk_results_combined
        }

# ======= Display Functions =======
def display_search_results(results: Dict[str, Any]):
    """Display search results in a formatted way."""
    
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS FOR: {results['query']}")
    print(f"{'='*80}")
    
    # Sub-intents
    print(f"\n🎯 SUB-INTENTS ({len(results['subintent_results'])}):")
    print("-" * 50)
    for i, result in enumerate(results['subintent_results'], 1):
        print(f"{i}. [{result['similarity_score']:.4f}] {result['title']}")
        print(f"   Description: {result['description'][:100]}...")
        print(f"   Sub-intent ID: {result['sub_intent_id']}")
        print()
    
    # Documents
    print(f"\n📄 DOCUMENTS ({len(results['document_results'])}):")
    print("-" * 50)
    for i, result in enumerate(results['document_results'], 1):
        print(f"{i}. [{result['similarity_score']:.4f}] {result['file_name']}")
        print(f"   Title: {result['title']}")
        print(f"   File ID: {result['file_id']}")
        print(f"   Intent: {result['intent']} -> {result['sub_intent']}")
        if result['full_summary']:
            print(f"   Summary: {result['full_summary'][:150]}...")
        print()
    
    # Chunks - Embedding
    print(f"\n📝 CHUNKS (EMBEDDING) ({len(results['chunk_results_embedding'])}):")
    print("-" * 50)
    for i, result in enumerate(results['chunk_results_embedding'], 1):
        print(f"{i}. [{result['similarity_score']:.4f}] Chunk {result['chunk_order']} from {result['file_id']}")
        print(f"   Title: {result['title']}")
        print(f"   Text: {result['chunk_text'][:200]}...")
        if result['summary']:
            print(f"   Summary: {result['summary'][:100]}...")
        print()
    
    # Chunks - Combined Embedding
    print(f"\n📝 CHUNKS (COMBINED EMBEDDING) ({len(results['chunk_results_combined'])}):")
    print("-" * 50)
    for i, result in enumerate(results['chunk_results_combined'], 1):
        print(f"{i}. [{result['similarity_score']:.4f}] Chunk {result['chunk_order']} from {result['file_id']}")
        print(f"   Title: {result['title']}")
        print(f"   Combined Context: {result['combined_context'][:200]}...")
        print()

# ======= Main Runner =======
async def main():
    """Main function to run the complete search pipeline."""
    
    # Configuration
    db_config = DatabaseConfig()
    api_key = os.getenv("google_api_key") or "your_google_api_key_here"
    
    if api_key == "your_google_api_key_here":
        logger.error("Please set your Google API key in the GOOGLE_API_KEY environment variable")
        return
    
    # Test queries
    test_queries = [
        "what is bill of materials in construction?",
    ]
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(api_key)
    search_pipeline = DocumentSearchPipeline(db_config, embedding_generator)
    
    # Run searches
    for query in test_queries:
        try:
            print(f"\n{'='*100}")
            print(f"PROCESSING QUERY: {query}")
            print(f"{'='*100}")
            
            results = await search_pipeline.search(
                query=query,
                subintent_top_k=3,
                document_top_k=5,
                chunk_top_k=10
            )
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
                continue
            
            display_search_results(results)
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            continue
        
        # Add delay between queries to respect API limits
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())