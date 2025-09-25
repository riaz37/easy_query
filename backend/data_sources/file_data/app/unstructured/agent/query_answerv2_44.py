import os
import asyncio
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
import json
from datetime import datetime

# Import the search pipeline from the existing file
# Assuming the previous file is named 'enhanced_search_pipeline.py'
from app.unstructured.agent.data_pull_v2 import (
    DatabaseConfig, 
    EmbeddingGenerator, 
    EnhancedQueryAnalyzer, 
    LLMReranker,
    DocumentSearchPipelineV2
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======= Enums for Configuration =======
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

# ======= Chunk Text Extractor =======
class ChunkTextExtractor:
    """Extracts chunk_text from database for given chunk IDs."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def extract_chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Extract chunk_text for given chunk IDs from document_chunks table."""
        if not chunk_ids:
            return {}
        
        sql = """
            SELECT chunk_id, chunk_text
            FROM document_chunks
            WHERE chunk_id = ANY(%s)
        """
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (chunk_ids,))
                    results = cursor.fetchall()
                    
                    chunk_texts = {}
                    for row in results:
                        chunk_texts[row['chunk_id']] = row['chunk_text'] or ""
                    
                    logger.info(f"Extracted chunk texts for {len(chunk_texts)}/{len(chunk_ids)} chunks")
                    return chunk_texts
                    
        except Exception as e:
            logger.error(f"Database error extracting chunk texts: {e}")
            return {}

# ======= Answer Generator =======
class AnswerGenerator:
    """Generates answers using LLM based on search results and extracted chunk texts."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-05-20"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def _prepare_context(self, chunks: List[Dict[str, Any]], chunk_texts: Dict[str, str], 
                        max_chunks: int) -> str:
        """Prepare context from chunks and their texts."""
        context_parts = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            chunk_id = chunk.get('chunk_id', '')
            chunk_text = chunk_texts.get(chunk_id, '')
            
            if chunk_text.strip():
                title = chunk.get('title', 'Unknown')
                summary = chunk.get('summary', 'No summary available')
                
                context_part = f"""
Document {i}:
Title: {title}
Summary: {summary}
Content: {chunk_text}
---
"""
                context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _get_answer_prompt(self, query: str, context: str, answer_style: AnswerStyle) -> str:
        """Generate the appropriate prompt based on answer style."""
        style_instructions = "Provide a detailed and comprehensive answer."

        base_prompt = f"""
You are an expert document analyst. Based on the provided documents, answer the user's question comprehensively and accurately.

User Question: {query}

Context Documents:
{context}

Instructions:
"""
        
        if answer_style == AnswerStyle.DETAILED:
            style_instructions = """
- Provide a comprehensive and detailed answer
- Include specific details and examples from the documents
- Explain the reasoning behind your conclusions
- If multiple perspectives exist, present them all
- Use clear structure with paragraphs for different aspects
"""
        elif answer_style == AnswerStyle.CONCISE:
            style_instructions = """
- Provide a brief and to-the-point answer
- Focus on the most important information
- Avoid unnecessary details
- Keep the response under 3 paragraphs
"""
        elif answer_style == AnswerStyle.BULLET_POINTS:
            style_instructions = """
- Structure your answer as clear bullet points
- Each point should be concise but informative
- Group related information together
- Use sub-bullets for detailed explanations if needed
"""
        elif answer_style == AnswerStyle.ANALYTICAL:
            style_instructions = """
- Provide an analytical response that breaks down the information
- Compare and contrast different aspects from the documents
- Identify patterns, trends, or relationships
- Draw insights and conclusions based on the evidence
- Structure your analysis logically
"""
        
        additional_instructions = """
- Only use information from the provided documents
- If the documents don't contain enough information to answer the question, clearly state this
- Cite specific documents when referencing information (e.g., "According to Document 1...")
- If there are contradictions between documents, highlight them
- Be objective and factual in your response
"""
        
        return base_prompt + style_instructions + additional_instructions + "\n\nAnswer:"
    
    async def generate_answer(self, query: str, chunks: List[Dict[str, Any]], 
                            chunk_texts: Dict[str, str], max_chunks: int = 10,
                            answer_style: AnswerStyle = AnswerStyle.DETAILED) -> Dict[str, Any]:
        """Generate answer based on query, chunks, and their texts."""
        
        if not chunks:
            return {
                "answer": "No relevant documents found to answer the query.",
                "sources_used": 0,
                "confidence": "low"
            }
        
        # Prepare context
        context = self._prepare_context(chunks, chunk_texts, max_chunks)
        
        if not context.strip():
            return {
                "answer": "No readable content found in the retrieved documents.",
                "sources_used": 0,
                "confidence": "low"
            }
        
        # Generate prompt
        prompt = self._get_answer_prompt(query, context, answer_style)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            answer = response.text.strip()
            sources_used = min(len(chunks), max_chunks)
            
            # Estimate confidence based on number of sources and content length
            confidence = "high" if sources_used >= 3 and len(answer) > 100 else "medium" if sources_used >= 1 else "low"
            
            return {
                "answer": answer,
                "sources_used": sources_used,
                "confidence": confidence,
                "context_length": len(context),
                "prompt_length": len(prompt)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources_used": 0,
                "confidence": "low"
            }

# ======= Enhanced Search and Answer Pipeline =======
class EnhancedSearchAnswerPipeline:
    """Complete pipeline that searches documents and generates answers."""
    
    def __init__(self, db_config: DatabaseConfig, api_key: str):
        self.db_config = db_config
        self.api_key = api_key
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(api_key)
        self.query_analyzer = EnhancedQueryAnalyzer(api_key)
        self.reranker = LLMReranker(api_key)
        self.search_pipeline = DocumentSearchPipelineV2(
            db_config, self.embedding_generator, self.query_analyzer, self.reranker
        )
        self.chunk_extractor = ChunkTextExtractor(db_config)
        self.answer_generator = AnswerGenerator(api_key)
    
    def _select_chunks_by_source(self, search_results: Dict[str, Any], 
                               chunk_source: ChunkSourceType) -> List[Dict[str, Any]]:
        """Select chunks based on the specified source type."""
        chunk_results = search_results.get('chunk_results', {})
        
        if chunk_source == ChunkSourceType.NORMAL:
            return chunk_results.get('normal', [])
        elif chunk_source == ChunkSourceType.FILTERED:
            return chunk_results.get('filtered', [])
        elif chunk_source == ChunkSourceType.OVERLAP:
            return chunk_results.get('overlap', [])
        elif chunk_source == ChunkSourceType.RERANKED:
            return chunk_results.get('reranked', [])
        else:
            return chunk_results.get('reranked', [])  # Default to reranked
    
    async def search_and_answer(self, 
                               query: str,
                               # Search pipeline parameters
                               use_intent_reranker: bool = True,
                               use_chunk_reranker: bool = True,
                               use_dual_embeddings: bool = True,
                               intent_top_k: int = 8,
                               chunk_top_k: int = 20,
                               # Answer generation parameters
                               chunk_source: ChunkSourceType = ChunkSourceType.RERANKED,
                               max_chunks_for_answer: int = 10,
                               answer_style: AnswerStyle = AnswerStyle.DETAILED) -> Dict[str, Any]:
        """
        Complete pipeline: search documents and generate answer.
        
        Args:
            query: The search query
            use_intent_reranker: Enable/disable intent re-ranking
            use_chunk_reranker: Enable/disable chunk re-ranking
            use_dual_embeddings: Enable/disable dual embedding search
            intent_top_k: Number of top intents to retrieve
            chunk_top_k: Number of top chunks to retrieve
            chunk_source: Which chunk source to use (normal, filtered, overlap, reranked)
            max_chunks_for_answer: Maximum number of chunks to use for answer generation
            answer_style: Style of answer generation (detailed, concise, bullet_points, analytical)
        
        Returns:
            Complete results including search results and generated answer
        """

        logger.info(f"Starting search and answer pipeline for query: {query[:50]}...")
        
        # Step 1: Perform search
        logger.info("Step 1: Performing document search...")
        search_results = await self.search_pipeline.search(
            query=query,
            use_intent_reranker=use_intent_reranker,
            use_chunk_reranker=use_chunk_reranker,
            use_dual_embeddings=use_dual_embeddings,
            intent_top_k=intent_top_k,
            chunk_top_k=chunk_top_k
        )
        
        if "error" in search_results:
            return {
                "query": query,
                "search_results": search_results,
                "answer_results": {
                    "answer": f"Search failed: {search_results['error']}",
                    "sources_used": 0,
                    "confidence": "low"
                },
                "pipeline_config": {
                    "search_config": {
                        "use_intent_reranker": use_intent_reranker,
                        "use_chunk_reranker": use_chunk_reranker,
                        "use_dual_embeddings": use_dual_embeddings,
                        "intent_top_k": intent_top_k,
                        "chunk_top_k": chunk_top_k
                    },
                    "answer_config": {
                        "chunk_source": chunk_source.value,
                        "max_chunks_for_answer": max_chunks_for_answer,
                        "answer_style": answer_style.value
                    }
                }
            }
        
        # Step 2: Select chunks based on source type
        logger.info(f"Step 2: Selecting chunks from source: {chunk_source.value}")
        selected_chunks = self._select_chunks_by_source(search_results, chunk_source)
        
        if not selected_chunks:
            return {
                "query": query,
                "search_results": search_results,
                "answer_results": {
                    "answer": f"No chunks available from source: {chunk_source.value}",
                    "sources_used": 0,
                    "confidence": "low"
                },
                "pipeline_config": {
                    "search_config": {
                        "use_intent_reranker": use_intent_reranker,
                        "use_chunk_reranker": use_chunk_reranker,
                        "use_dual_embeddings": use_dual_embeddings,
                        "intent_top_k": intent_top_k,
                        "chunk_top_k": chunk_top_k
                    },
                    "answer_config": {
                        "chunk_source": chunk_source.value,
                        "max_chunks_for_answer": max_chunks_for_answer,
                        "answer_style": answer_style.value
                    }
                }
            }
        
        # Step 3: Extract chunk texts
        logger.info("Step 3: Extracting chunk texts from database...")
        chunk_ids = [chunk.get('chunk_id', '') for chunk in selected_chunks if chunk.get('chunk_id')]
        chunk_texts = self.chunk_extractor.extract_chunk_texts(chunk_ids)
        
        # Step 4: Generate answer
        logger.info("Step 4: Generating answer using LLM...")
        answer_results = await self.answer_generator.generate_answer(
            query=query,
            chunks=selected_chunks,
            chunk_texts=chunk_texts,
            max_chunks=max_chunks_for_answer,
            answer_style=answer_style
        )
        
        # Compile final results
        final_results = {
            "query": query,
            "search_results": search_results,
            "answer_results": answer_results,
            "pipeline_config": {
                "search_config": {
                    "use_intent_reranker": use_intent_reranker,
                    "use_chunk_reranker": use_chunk_reranker,
                    "use_dual_embeddings": use_dual_embeddings,
                    "intent_top_k": intent_top_k,
                    "chunk_top_k": chunk_top_k
                },
                "answer_config": {
                    "chunk_source": chunk_source.value,
                    "max_chunks_for_answer": max_chunks_for_answer,
                    "answer_style": answer_style.value
                }
            },
            "processing_stats": {
                "total_chunks_found": len(selected_chunks),
                "chunks_with_text": len(chunk_texts),
                "chunks_used_for_answer": answer_results.get('sources_used', 0)
            }
        }
        
        logger.info("Search and answer pipeline completed successfully!")
        return final_results


# ======= Display Functions =======
def display_answer_results(results: Dict[str, Any]):
    """Display the complete search and answer results."""
    
    print(f"\n{'='*100}")
    print(f"SEARCH & ANSWER RESULTS FOR: {results['query']}")
    print(f"{'='*100}")
    
    # Configuration
    search_config = results['pipeline_config']['search_config']
    answer_config = results['pipeline_config']['answer_config']
    
    print(f"\n🔧 PIPELINE CONFIGURATION:")
    print("-" * 50)
    print(f"Search Configuration:")
    print(f"  - Intent Re-ranker: {'✅' if search_config['use_intent_reranker'] else '❌'}")
    print(f"  - Chunk Re-ranker: {'✅' if search_config['use_chunk_reranker'] else '❌'}")
    print(f"  - Dual Embeddings: {'✅' if search_config['use_dual_embeddings'] else '❌'}")
    print(f"  - Intent Top-K: {search_config['intent_top_k']}")
    print(f"  - Chunk Top-K: {search_config['chunk_top_k']}")
    
    print(f"\nAnswer Configuration:")
    print(f"  - Chunk Source: {answer_config['chunk_source'].upper()}")
    print(f"  - Max Chunks: {answer_config['max_chunks_for_answer']}")
    print(f"  - Answer Style: {answer_config['answer_style'].upper()}")
    
    # Processing Statistics
    stats = results.get('processing_stats', {
        "total_chunks_found": "N/A",
        "chunks_with_text": "N/A",
        "chunks_used_for_answer": "N/A"
    })

    print(f"\n📊 PROCESSING STATISTICS:")
    print("-" * 50)
    print(f"Total chunks found: {stats['total_chunks_found']}")
    print(f"Chunks with text: {stats['chunks_with_text']}")
    print(f"Chunks used for answer: {stats['chunks_used_for_answer']}")
    
    # Answer Results
    answer_results = results['answer_results']
    print(f"\n🎯 GENERATED ANSWER:")
    print("-" * 50)
    print(f"Confidence: {answer_results.get('confidence', 'N/A').upper()}")
    print(f"Sources Used: {answer_results.get('sources_used', 0)}")
    print(f"Context Length: {answer_results.get('context_length', 0)} characters")
    print(f"\nAnswer:")
    print("-" * 30)
    print(answer_results.get('answer', 'No answer generated'))
    
    # Search Results Summary
    search_results = results['search_results']
    if 'stats' in search_results:
        search_stats = search_results['stats']
        print(f"\n🔍 SEARCH RESULTS SUMMARY:")
        print("-" * 50)
        print(f"Intents kept: {search_stats.get('intents_kept', 'N/A')}")
        print(f"Chunks kept: {search_stats.get('chunks_kept', 'N/A')}")
        
        if 'dual_embedding_stats' in search_stats:
            dual_stats = search_stats['dual_embedding_stats']
            print(f"Regular embedding chunks: {dual_stats['regular_chunks']}")
            print(f"Combined embedding chunks: {dual_stats['combined_chunks']}")
            print(f"Merged chunks: {dual_stats['merged_chunks']}")
    
    print(f"\n{'='*100}")


# ======= Main Execution Function =======
async def main():
    """Main function to demonstrate the enhanced search and answer pipeline."""
    
    # Configuration
    API_KEY = os.getenv("google_api_key")
    if not API_KEY:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return
    
    db_config = DatabaseConfig()
    
    # Initialize pipeline
    print("🔧 Initializing search and answer pipeline...")
    pipeline = EnhancedSearchAnswerPipeline(db_config, API_KEY)
    
    # Test queries
    test_queries = [
        "How much did IFIC earn from debit card fees in 2022, and what was the percentage increase compared to the previous year?"
    ]
    
    # Different configurations to test
    test_configurations = [
        {
            "name": "Standard Configuration",
            "config": {
                "use_intent_reranker": True,
                "use_chunk_reranker": True,
                "use_dual_embeddings": True,
                "intent_top_k": 10,
                "chunk_top_k": 40,
                "chunk_source": ChunkSourceType.RERANKED,
                "max_chunks_for_answer": 40,
                "answer_style": AnswerStyle.DETAILED
            }
        },
        {
            "name": "Concise Answers",
            "config": {
                "use_intent_reranker": True,
                "use_chunk_reranker": True,
                "use_dual_embeddings": False,
                "intent_top_k": 10,
                "chunk_top_k": 40,
                "chunk_source": ChunkSourceType.RERANKED,
                "max_chunks_for_answer": 40,
                "answer_style": AnswerStyle.CONCISE
            }
        },
        {
            "name": "Bullet Points Style",
            "config": {
                "use_intent_reranker": True,
                "use_chunk_reranker": True,
                "use_dual_embeddings": True,
                "intent_top_k": 10,
                "chunk_top_k": 40,
                "chunk_source": ChunkSourceType.OVERLAP,
                "max_chunks_for_answer": 40,
                "answer_style": AnswerStyle.BULLET_POINTS
            }
        }
    ]
    
    print("🚀 Starting search and answer pipeline tests...")
    print(f"{'='*100}")
    
    for query_idx, query in enumerate(test_queries, 1):
        print(f"\n🔍 TESTING QUERY {query_idx}/{len(test_queries)}: {query}")
        print("=" * 80)
        
        for config_idx, test_config in enumerate(test_configurations, 1):
            print(f"\n📋 Configuration {config_idx}: {test_config['name']}")
            print("-" * 60)
            
            try:
                results = await pipeline.search_and_answer(
                    query=query,
                    **test_config['config']
                )
                
                display_answer_results(results)
                
                # Add separator between configurations
                if config_idx < len(test_configurations):
                    print(f"\n{'-'*50}")
                    
            except Exception as e:
                print(f"❌ Error processing query '{query}' with config '{test_config['name']}': {e}")
                logger.error(f"Error in main execution: {e}")
    
    print(f"\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())