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

from dotenv import load_dotenv
load_dotenv(override=True)

BASE_URL= os.getenv("BASE_URL", "http://localhost:8902")

# Import the search pipeline from the existing file
# Assuming the previous file is named 'enhanced_search_pipeline.py'
from data_sources.file_data.app.unstructured.agent.data_pull_v2  import (
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

# ======= Enhanced Chunk Text Extractor =======
class ChunkTextExtractor:
    """Extracts chunk_text and metadata from database for given chunk IDs."""
    
    def __init__(self, db_config: DatabaseConfig, uploads_base_path: str = None):
        self.db_config = db_config
        # Use provided path or default to your absolute path
        if uploads_base_path is None:
            self.uploads_base_path = "uploads"
        else:
            self.uploads_base_path = uploads_base_path
    
    def _format_page_range(self, page_numbers: List[int]) -> str:
        """Format page numbers into a range string."""
        if not page_numbers:
            return "N/A"
        
        # Sort page numbers
        sorted_pages = sorted(page_numbers)
        
        if len(sorted_pages) == 1:
            return str(sorted_pages[0])
        elif len(sorted_pages) == 2:
            return f"{sorted_pages[0]}-{sorted_pages[1]}"
        else:
            return f"{sorted_pages[0]}-{sorted_pages[-1]}"
    
    def _get_file_path(self, file_name: str) -> str:
        """Get full file path based on file name, ignoring extension."""
        if not file_name or file_name == 'Unknown File':
            return "No file found in the path"
        
        base_name = os.path.splitext(file_name)[0]  # remove extension if any
        
        # Check files in the uploads directory
        try:
            for existing_file in os.listdir(self.uploads_base_path):
                existing_base_name, _ = os.path.splitext(existing_file)
                if existing_base_name == base_name:
                    full_path = os.path.join(self.uploads_base_path, existing_file)
                    return BASE_URL+full_path
        except Exception as e:
            logger.error(f"Error accessing uploads directory: {e}")
            return "No file found in the path"
        
        return "No file found in the path"

    
    def extract_chunk_texts_with_metadata(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract chunk_text and metadata for given chunk IDs from intent_chunks table."""
        if not chunk_ids:
            return {}
        
        sql = """
            SELECT chunk_id, chunk_text,metadata
            FROM document_chunks
            WHERE chunk_id = ANY(%s)
        """
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(sql, (chunk_ids,))
                    results = cursor.fetchall()
                    
                    chunk_data = {}
                    for row in results:
                        chunk_id = row['chunk_id']
                        chunk_text = row['chunk_text'] or ""
                        metadata = row['metadata'] or {}
                        
                        # Extract file name and page range from metadata
                        file_name = metadata.get('file_name', 'Unknown File')
                        page_numbers = metadata.get('page_numbers', [])
                        page_range = self._format_page_range(page_numbers)
                        
                        # Get file path
                        file_path = self._get_file_path(file_name)
                        
                        chunk_data[chunk_id] = {
                            'chunk_text': chunk_text,
                            'file_name': file_name,
                            'file_path': file_path,  # New field added
                            'page_range': page_range,
                            'metadata': metadata
                        }
                    
                    logger.info(f"Extracted chunk data with metadata for {len(chunk_data)}/{len(chunk_ids)} chunks")
                    return chunk_data
                    
        except Exception as e:
            logger.error(f"Database error extracting chunk texts with metadata: {e}")
            return {}
    def extract_chunk_texts(self, chunk_ids: List[str]) -> Dict[str, str]:
        """Extract chunk_text only for backward compatibility."""
        chunk_data = self.extract_chunk_texts_with_metadata(chunk_ids)
        return {chunk_id: data['chunk_text'] for chunk_id, data in chunk_data.items()}


        
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")


# ======= Enhanced Answer Generator =======
class AnswerGenerator:
    """Generates answers using LLM based on search results and extracted chunk texts."""
    
    def __init__(self, api_key: str, model_name: str = gemini_model_name):
        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            temperature=0
        )
        self.model = genai.GenerativeModel(model_name,generation_config=generation_config)
    
    def _prepare_context_with_sources(self, chunks: List[Dict[str, Any]], 
                                    chunk_data: Dict[str, Dict[str, Any]], 
                                    max_chunks: int) -> Tuple[str, List[Dict[str, str]]]:
        """Prepare context from chunks and their texts with source information."""
        context_parts = []
        sources_used = []
        
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            chunk_id = chunk.get('chunk_id', '')
            chunk_info = chunk_data.get(chunk_id, {})
            chunk_text = chunk_info.get('chunk_text', '')
            
            if chunk_text.strip():
                title = chunk.get('title', 'Unknown')
                summary = chunk.get('summary', 'No summary available')
                file_name = chunk_info.get('file_name', 'Unknown File')
                file_path = chunk_info.get('file_path', 'No file found in the path')  # New field
                page_range = chunk_info.get('page_range', 'N/A')
                
                # Add source information
                source_info = {
                    'document_number': i,
                    'file_name': file_name,
                    'file_path': file_path,  # New field added
                    'page_range': page_range,
                    'title': title
                }
                sources_used.append(source_info)
                
                context_part = f"""
Document {i}:
Title: {title}
Source: {file_name} (Pages: {page_range})
File Path: {file_path}
Summary: {summary}
Content: {chunk_text}
---
"""
                context_parts.append(context_part)
        
        return "\n".join(context_parts), sources_used
    def _get_answer_prompt_with_sources(self, query: str, context: str, 
                                      answer_style: AnswerStyle) -> str:
        """Generate the appropriate prompt based on answer style with source citation requirements."""
        style_instructions = "Provide a detailed and comprehensive answer."
        base_prompt = f"""# Source Analysis Task

You are an expert source analyst. Based on the provided sources, answer the user's question comprehensively and accurately using proper markdown formatting.

## User Question
**{query}**

## Context Sources
{context}

---

## Response Requirements
"""

        # Define answer style instructions
        if answer_style == AnswerStyle.DETAILED:
            style_instructions = """### Answer Style: Detailed Analysis
    - Provide a **comprehensive and detailed answer** using proper markdown formatting
    - Include specific details and examples from the sources
    - Explain the reasoning behind your conclusions
    - If multiple perspectives exist, present them all
    - Use clear structure with headings (`##`, `###`, `####`) and paragraphs for different aspects
    - Utilize **bold** for key points and *italics* for emphasis where appropriate
    - Include well-formatted tables for complex information
    - Use `code formatting` for technical terms, specific values, and data points
    """
            
        elif answer_style == AnswerStyle.CONCISE:
            style_instructions = """### Answer Style: Concise Summary
    - Provide a **brief and to-the-point answer** using markdown formatting
    - Focus on the most important information
    - Avoid unnecessary details
    - Keep the response under 3 paragraphs
    - Use **bold** to highlight critical information
    - Use `code formatting` for specific terms and values
    - Maintain clarity while being succinct
    """
            
        elif answer_style == AnswerStyle.BULLET_POINTS:
            style_instructions = """### Answer Style: Structured Bullet Points
    - Structure your answer using **markdown bullet points** (`-` or `*`)
    - Each point should be concise but informative
    - Group related information together under appropriate headings
    - Use sub-bullets (indented) for detailed explanations if needed
    - Apply **bold** formatting for key terms and concepts
    - Use `code formatting` for technical terms and specific values
    - Maintain logical flow and organization
    """
            
        elif answer_style == AnswerStyle.ANALYTICAL:
            style_instructions = """### Answer Style: Analytical Breakdown
    - Provide an **analytical response** using structured markdown formatting
    - Compare and contrast different aspects from the sources using well-formatted tables
    - Identify patterns, trends, or relationships
    - Draw insights and conclusions based on the evidence
    - Structure your analysis with clear headings (`##`, `###`, `####`) and subheadings
    - Use **bold** for key findings and *italics* for supporting evidence
    - Use `code formatting` for technical terms and data points
    - Include synthesis and interpretation of information
    """

        additional_instructions = """---

    ## Markdown Formatting Standards

    ### Structure Hierarchy
    - `#` for main title
    - `##` for major sections  
    - `###` for subsections
    - `####` for sub-subsections if needed

    ### Text Formatting Guidelines
    - **Bold** for: key terms, important findings, critical information
    - *Italics* for: emphasis, supporting details, clarifications
    - `Code formatting` for: technical terms, specific values, data points, names, email addresses

    ### Organization Elements
    - Bullet points (`-`) for lists and key points
    - Numbered lists (`1.`) for sequential information or steps
    - Horizontal rules (`---`) to separate major sections
    - Block quotes (`>`) for direct citations and important callouts

    ### Table Formatting Requirements
    Tables should be well-formatted with proper alignment:

    ```markdown
    | Header 1 | Header 2 | Header 3 |
    |----------|----------|----------|
    | Data 1   | Data 2   | Data 3   |
    | Data 4   | Data 5   | Data 6   |
    ```

    ### Visual Enhancement
    - Use tables for complex comparative information
    - Include callout boxes for warnings or important notes
    - Apply consistent formatting throughout the response
    - Ensure proper spacing and readability

    ---

    ## Content Guidelines

    ### Information Usage
    - **ONLY** use information from the provided sources
    - Do not introduce external knowledge or assumptions
    - Stay strictly within the bounds of the provided material

    ### Insufficient Information Handling
    When sources don't contain enough information:
    > ⚠️ **Note**: The provided sources do not contain sufficient information to fully answer this question.

    ### Contradiction Management
    When sources contradict each other:

    #### Source Contradictions
    - Document A states: [information]
    - Document B contradicts with: [conflicting information]
    - **Analysis**: [Your interpretation of the contradiction]

    ### Objectivity Requirements
    - Be **objective and factual** in your response
    - Present information without bias
    - Let the sources speak for themselves
    - Distinguish between facts and interpretations

    ---

    ## Source Citation Requirements

    ### Important Citation Rules
    - **DO NOT** mention sources within the text body
    - **DO NOT** include source references in paragraphs
    - **DO NOT** use phrases like "According to Source 1" or "The document states"
    - Write as if the information is factual without attributing it mid-text

    ### Final Source Attribution
    At the end of your response, include:

    ```markdown
    ---

    **SOURCES:** document 1, document 2, document 3
    ```

    ### Direct Quotes Format
    When using direct quotes:
    ```markdown
    > "Direct quote from source"
    ```

    ---

    **Begin your markdown-formatted analysis below:**
    """
        return base_prompt + style_instructions + additional_instructions

    async def generate_answer_with_sources(self, query: str, chunks: List[Dict[str, Any]], 
                                         chunk_data: Dict[str, Dict[str, Any]], 
                                         max_chunks: int = 10,
                                         answer_style: AnswerStyle = AnswerStyle.DETAILED) -> Dict[str, Any]:
        """Generate answer with source information based on query, chunks, and their texts."""
        
        if not chunks:
            return {
                "answer": "No relevant documents found to answer the query.",
                "sources_used": 0,
                "confidence": "low",
                "sources": []
            }
        
        # Prepare context with sources
        context, sources_used = self._prepare_context_with_sources(chunks, chunk_data, max_chunks)
        
        if not context.strip():
            return {
                "answer": "No readable content found in the retrieved documents.",
                "sources_used": 0,
                "confidence": "low",
                "sources": []
            }
        
        # Generate prompt
        prompt = self._get_answer_prompt_with_sources(query, context, answer_style)
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            answer = response.text.strip()
            num_sources = len(sources_used)
            
            # Estimate confidence based on number of sources and content length
            confidence = "high" if num_sources >= 3 and len(answer) > 100 else "medium" if num_sources >= 1 else "low"
            
            return {
                "answer": answer,
                "sources_used": num_sources,
                "confidence": confidence,
                "context_length": len(context),
                "prompt_length": len(prompt),
                "sources": sources_used
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources_used": 0,
                "confidence": "low",
                "sources": []
            }
    
    # Keep backward compatibility
    async def generate_answer(self, query: str, chunks: List[Dict[str, Any]], 
                            chunk_texts: Dict[str, str], max_chunks: int = 10,
                            answer_style: AnswerStyle = AnswerStyle.DETAILED) -> Dict[str, Any]:
        """Backward compatibility method."""
        # Convert chunk_texts to chunk_data format
        chunk_data = {}
        for chunk_id, text in chunk_texts.items():
            chunk_data[chunk_id] = {
                'chunk_text': text,
                'file_name': 'Unknown File',
                'page_range': 'N/A',
                'metadata': {}
            }
        
        return await self.generate_answer_with_sources(query, chunks, chunk_data, max_chunks, answer_style)

# ======= Enhanced Search and Answer Pipeline =======
class EnhancedSearchAnswerPipeline:
    """Complete pipeline that searches documents and generates answers with sources."""
    
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
                               answer_style: AnswerStyle = AnswerStyle.DETAILED,
                               # Table-specific parameters
                               table_specific: bool = False,
                               tables: List[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: search documents and generate answer with sources.
        
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
            table_specific: If True, search only in specified tables
            tables: List of table names to search in when table_specific is True
        
        Returns:
            Complete results including search results and generated answer with sources
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
            chunk_top_k=chunk_top_k,
            table_specific=table_specific,
            tables=tables
        )
        
        if "error" in search_results:
            return {
                "query": query,
                "search_results": search_results,
                "answer_results": {
                    "answer": f"Search failed: {search_results['error']}",
                    "sources_used": 0,
                    "confidence": "low",
                    "sources": []
                },
                "pipeline_config": {
                    "search_config": {
                        "use_intent_reranker": use_intent_reranker,
                        "use_chunk_reranker": use_chunk_reranker,
                        "use_dual_embeddings": use_dual_embeddings,
                        "intent_top_k": intent_top_k,
                        "chunk_top_k": chunk_top_k,
                        "table_specific": table_specific,
                        "tables": tables
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
                    "confidence": "low",
                    "sources": []
                },
                "pipeline_config": {
                    "search_config": {
                        "use_intent_reranker": use_intent_reranker,
                        "use_chunk_reranker": use_chunk_reranker,
                        "use_dual_embeddings": use_dual_embeddings,
                        "intent_top_k": intent_top_k,
                        "chunk_top_k": chunk_top_k,
                        "table_specific": table_specific,
                        "tables": tables
                    },
                    "answer_config": {
                        "chunk_source": chunk_source.value,
                        "max_chunks_for_answer": max_chunks_for_answer,
                        "answer_style": answer_style.value
                    }
                }
            }
        
        # Step 3: Extract chunk texts with metadata
        logger.info("Step 3: Extracting chunk texts with metadata from database...")
        chunk_ids = [chunk.get('chunk_id', '') for chunk in selected_chunks if chunk.get('chunk_id')]
        chunk_data = self.chunk_extractor.extract_chunk_texts_with_metadata(chunk_ids)
        
        # Step 4: Generate answer with sources
        logger.info("Step 4: Generating answer with sources using LLM...")
        answer_results = await self.answer_generator.generate_answer_with_sources(
            query=query,
            chunks=selected_chunks,
            chunk_data=chunk_data,
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
                    "chunk_top_k": chunk_top_k,
                    "table_specific": table_specific,
                    "tables": tables
                },
                "answer_config": {
                    "chunk_source": chunk_source.value,
                    "max_chunks_for_answer": max_chunks_for_answer,
                    "answer_style": answer_style.value
                }
            },
            "processing_stats": {
                "total_chunks_found": len(selected_chunks),
                "chunks_with_text": len(chunk_data),
                "chunks_used_for_answer": answer_results.get('sources_used', 0)
            }
        }
        
        logger.info("Search and answer pipeline completed successfully!")
        return final_results


# ======= Enhanced Display Functions =======
def display_answer_results(results: Dict[str, Any]):
    """Display the complete search and answer results with sources."""
    
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
    print(f"  - Table Specific: {'✅' if search_config.get('table_specific', False) else '❌'}")
    if search_config.get('table_specific', False) and search_config.get('tables'):
        print(f"  - Tables: {', '.join(search_config['tables'])}")
    
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
    
    # Sources Information
    sources = answer_results.get('sources', [])
    if sources:
        print(f"\n📚 SOURCES USED:")
        print("-" * 50)
        for source in sources:
            print(f"Document {source['document_number']}:")
            print(f"  📄 File: {source['file_name']}")
            print(f"  📖 Pages: {source['page_range']}")
            print(f"  📝 Title: {source['title']}")
            print()
    
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
    """Main function to demonstrate the enhanced search and answer pipeline with sources."""
    
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
        "tell me requirments for tech stack"
    ]
    
    # Different configurations to test
    test_configurations = [
        {
            "name": "Standard Configuration",
            "config": {
                "use_intent_reranker": False,
                "use_chunk_reranker": False,
                "use_dual_embeddings": True,
                "intent_top_k": 10,
                "chunk_top_k": 40,
                "chunk_source": ChunkSourceType.RERANKED,
                "max_chunks_for_answer": 40,
                "answer_style": AnswerStyle.DETAILED
            }
        }
        # {
        #     "name": "Concise Answers",
        #     "config": {
        #         "use_intent_reranker": True,
        #         "use_chunk_reranker": True,
        #         "use_dual_embeddings": False,
        #         "intent_top_k": 10,
        #         "chunk_top_k": 40,
        #         "chunk_source": ChunkSourceType.RERANKED,
        #         "max_chunks_for_answer": 40,
        #         "answer_style": AnswerStyle.CONCISE
        #     }
        # },
        # {
        #     "name": "Bullet Points Style",
        #     "config": {
        #         "use_intent_reranker": True,
        #         "use_chunk_reranker": True,
        #         "use_dual_embeddings": True,
        #         "intent_top_k": 10,
        #         "chunk_top_k": 40,
        #         "chunk_source": ChunkSourceType.OVERLAP,
        #         "max_chunks_for_answer": 40,
        #         "answer_style": AnswerStyle.BULLET_POINTS
        #     }
        # }
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