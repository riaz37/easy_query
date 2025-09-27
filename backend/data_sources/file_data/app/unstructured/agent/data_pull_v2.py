import os
import asyncio
import logging
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import google.generativeai as genai
import numpy as np
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Retrieve API key and model name from environment variables
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")

google_gemini_embedding_name= os.getenv("google_gemini_embedding_name", "gemini-embedding-exp-03-07")
if google_gemini_embedding_name is None:    
    print("Warning: 'google_gemini_embedding_name' not found in environment variables, using default 'gemini-embedding-exp-03-07'.")

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

# ======= Enhanced Query Analyzer =======

class EnhancedQueryAnalyzer:
    """Analyzes queries using LLM to extract keywords and dates."""
    
    def __init__(self, api_key: str, model_name: str = gemini_model_name):
        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            temperature=0        )
        self.model = genai.GenerativeModel(model_name, generation_config=generation_config)
        
    def _extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON content from various text formats."""
        # Remove markdown code blocks
        text = re.sub(r'```(?:json)?\n?', '', text)
        text = text.strip()
        
        # Try to find JSON object boundaries
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return text
    
    def _validate_and_fix_json(self, json_str: str) -> Dict[str, Any]:
        """Attempt to parse and fix common JSON issues."""
        try:
            # First, try direct parsing
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting to fix...")
            
            # Common fixes
            fixes = [
                # Fix trailing commas
                lambda x: re.sub(r',\s*}', '}', x),
                lambda x: re.sub(r',\s*]', ']', x),
                # Fix single quotes to double quotes
                lambda x: re.sub(r"'([^']*)':", r'"\1":', x),
                lambda x: re.sub(r":\s*'([^']*)'", r': "\1"', x),
                # Fix unquoted keys
                lambda x: re.sub(r'(\w+):', r'"\1":', x),
                # Fix boolean values
                lambda x: x.replace('True', 'true').replace('False', 'false'),
            ]
            
            current_json = json_str
            for fix in fixes:
                try:
                    current_json = fix(current_json)
                    return json.loads(current_json)
                except json.JSONDecodeError:
                    continue
            
            # If all fixes fail, return default structure
            logger.error("Could not fix JSON, returning default structure")
            return self._get_default_response()
    
    def _get_default_response(self) -> Dict[str, Any]:
        """Return default response structure when parsing fails."""
        return {
            "keywords": [],
            "date_range": None,
            "filter": False
        }
    
    def _validate_response_structure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure response has correct structure."""
        validated = {
            "keywords": [],
            "date_range": None,
            "filter": False
        }
        
        # Validate keywords
        if "keywords" in response and isinstance(response["keywords"], list):
            validated["keywords"] = [str(k) for k in response["keywords"] if k]
        
        # Validate date_range
        if "date_range" in response:
            date_range = response["date_range"]
            if date_range is not None and isinstance(date_range, dict):
                if "start" in date_range and "end" in date_range:
                    validated["date_range"] = {
                        "start": str(date_range["start"]),
                        "end": str(date_range["end"])
                    }
        
        # Validate filter
        if "filter" in response:
            validated["filter"] = bool(response["filter"])
        
        return validated
        
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract keywords, dates, and filtering decision."""
        
        prompt = f"""
You are a smart document analysis assistant.
Given the following query, perform these tasks:
1. Extract ONLY significant keywords that would be useful for filtering documents. 
   - Include: names, organizations, places, important concepts, unique terms, technical terms
   - Exclude: common words, articles, prepositions, generic terms
   - Only include keywords that would meaningfully narrow down search results
2. Detect any date mentions. If dates are present, identify the earliest and latest dates in format YYYY-MM-DD. If no dates, return null.
3. Determine if keyword or date filtering will be helpful based on specificity and context.

Return ONLY valid JSON in this exact format (no extra text):
{{
  "keywords": ["keyword1", "keyword2"],
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
  "filter": true
}}

Query: {query}
"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, 
                prompt
            )
            
            response_text = response.text.strip()
            logger.debug(f"Raw LLM response: {response_text}")
            
            # Extract and clean JSON
            json_text = self._extract_json_from_text(response_text)
            
            # Parse and validate JSON
            analysis = self._validate_and_fix_json(json_text)
            analysis = self._validate_response_structure(analysis)
            
            logger.info(f"Query analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return self._get_default_response()
# ======= LLM Re-ranker =======
class LLMReranker:
    """Re-ranks search results using LLM to remove irrelevant items and reorder."""
    
    def __init__(self, api_key: str, model_name: str = gemini_model_name):
        genai.configure(api_key=api_key)
        generation_config = genai.GenerationConfig(
            temperature=0
        )
        self.model = genai.GenerativeModel(model_name,generation_config=generation_config)
    
    async def rerank_intents(self, query: str, intents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank intents based on relevance to query."""
        
        if not intents:
            return []
        
        # Prepare data for LLM
        intent_data = []
        for i, intent in enumerate(intents):
            intent_data.append({
                "index": i,
                "title": intent.get('title', ''),
                "description": intent.get('description', ''),
            })
        
        prompt = f"""
You are a document relevance expert. Given a user query and a list of intents, your task is to:

1. Remove intents that are NOT relevant to the query
2. Reorder the remaining intents by relevance (most relevant first)
3. Keep only intents that would genuinely help answer the query

Query: "{query}"

Intents:
{json.dumps(intent_data, indent=2)}

Return ONLY a JSON array of indices (original positions) in order of relevance:
[0, 2, 1] (example - keep only relevant ones, reordered by relevance)

If no intents are relevant, return: []
"""
        
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            # Extract JSON array
            if "[" in response_text and "]" in response_text:
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                json_text = response_text[json_start:json_end]
                indices = json.loads(json_text)
                
                # Return reordered intents
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(intents):
                        reordered_item = intents[idx].copy()
                        reordered_item['rerank_position'] = len(reranked) + 1
                        reordered_item['original_position'] = idx + 1
                        reranked.append(reordered_item)
                
                logger.info(f"Reranked {len(reranked)}/{len(intents)} intents")
                return reranked
            
        except Exception as e:
            logger.error(f"Failed to rerank intents: {e}")
        
        return intents  # Return original if reranking fails
    
    async def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query."""
        
        if not chunks:
            return []
        
        # Prepare data for LLM - include title and summary
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "index": i,
                "title": chunk.get('title', ''),
                "summary": chunk.get('summary', ''),
                "content_preview": chunk.get('combined_context', '')[:100] + "..." if chunk.get('combined_context') else ""
            })
        
        prompt = f"""
You are a document relevance expert. Given a user query and a list of text chunks, your task is to:

1. Remove chunks that are NOT relevant to the query
2. Reorder the remaining chunks by relevance (most relevant first)
3. Keep only chunks that would genuinely help answer the query

Query: "{query}"

Chunks:
{json.dumps(chunk_data, indent=2)}

Return ONLY a JSON array of indices (original positions) in order of relevance:
[0, 2, 1] (example - keep only relevant ones, reordered by relevance)

If no chunks are relevant, return: []
"""
        
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            response_text = response.text.strip()
            
            # Extract JSON array
            if "[" in response_text and "]" in response_text:
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                json_text = response_text[json_start:json_end]
                indices = json.loads(json_text)
                
                # Return reordered chunks
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(chunks):
                        reordered_item = chunks[idx].copy()
                        reordered_item['rerank_position'] = len(reranked) + 1
                        reordered_item['original_position'] = idx + 1
                        reranked.append(reordered_item)
                
                logger.info(f"Reranked {len(reranked)}/{len(chunks)} chunks")
                return reranked
            
        except Exception as e:
            logger.error(f"Failed to rerank chunks: {e}")
        
        return chunks  # Return original if reranking fails

# ======= Embedding Generator =======
class EmbeddingGenerator:
    """Generates embeddings using Google Generative AI embeddings API."""

    def __init__(self, api_key: str, model_name: str = google_gemini_embedding_name, output_dim: int = 1536):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.output_dim = output_dim

    async def generate_embedding(self, text: str, task_type: str = "RETRIEVAL_QUERY", retries: int = 3, delay: float = 1.5) -> Optional[List[float]]:
        """Generate embedding for a given text."""
        for attempt in range(retries):
            try:
                if attempt > 0:
                    await asyncio.sleep(delay * attempt)

                result = await asyncio.to_thread(
                    genai.embed_content,
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

# ======= Enhanced Vector Search Functions with Dual Embeddings =======
class EnhancedVectorSearchEngine:
    """Enhanced vector search with dual embedding support and keyword/date filtering."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def search_similar_intents(self, query_embedding: List[float], 
                             keywords: List[str] = None,
                             date_range: Dict[str, str] = None,
                             top_k: int = 10,
                             table_specific: bool = False,
                             tables: List[str] = None) -> Tuple[List[Dict], List[Dict]]:
        """Search for similar intents - returns (normal_results, filtered_results)."""
        
        # Normal vector search
        if table_specific and tables:
            # Join with document_chunks and document_files to filter by table
            normal_sql = """
                SELECT DISTINCT
                    ic.intent_id, 
                    ic.title, 
                    ic.description, 
                    ic.created_at,
                    ic.updated_at,
                    ic.embedding <=> %s::vector AS similarity_score
                FROM intent_chunks ic
                INNER JOIN document_chunks dc ON ic.intent_id = dc.intent_id
                INNER JOIN document_files df ON dc.file_id = df.file_id
                WHERE df.table_name = ANY(%s::text[])
                ORDER BY ic.embedding <=> %s::vector
                LIMIT %s;
            """
            normal_params = [query_embedding, tables, query_embedding, top_k]
        else:
            normal_sql = """
                SELECT 
                    intent_id, 
                    title, 
                    description, 
                    created_at,
                    updated_at,
                    embedding <=> %s::vector AS similarity_score
                FROM intent_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            normal_params = [query_embedding, query_embedding, top_k]
        
        # Filtered search with keywords
        filter_conditions = []
        filter_params = [query_embedding]
        
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                if table_specific and tables:
                    # Use table alias when joining tables
                    keyword_conditions.append("(ic.title ILIKE %s OR ic.description ILIKE %s)")
                else:
                    # Use unqualified names when not joining
                    keyword_conditions.append("(title ILIKE %s OR description ILIKE %s)")
                filter_params.extend([f'%{keyword}%', f'%{keyword}%'])
            
            if keyword_conditions:
                filter_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        if date_range and date_range.get('start') and date_range.get('end'):
            if table_specific and tables:
                # Use table alias when joining tables
                filter_conditions.append("""
                    (ic.created_at::date >= %s::date AND ic.created_at::date <= %s::date)
                """)
            else:
                # Use unqualified names when not joining
                filter_conditions.append("""
                    (created_at::date >= %s::date AND created_at::date <= %s::date)
                """)
            filter_params.extend([date_range['start'], date_range['end']])
        
        if table_specific and tables:
            # Add table filtering to filtered search
            if filter_conditions:
                filtered_sql = f"""
                    SELECT DISTINCT
                        ic.intent_id, 
                        ic.title, 
                        ic.description, 
                        ic.created_at,
                        ic.updated_at,
                        ic.embedding <=> %s::vector AS similarity_score
                    FROM intent_chunks ic
                    INNER JOIN document_chunks dc ON ic.intent_id = dc.intent_id
                    INNER JOIN document_files df ON dc.file_id = df.file_id
                    WHERE df.table_name = ANY(%s::text[]) AND {' AND '.join(filter_conditions)}
                    ORDER BY ic.embedding <=> %s::vector
                    LIMIT %s;
                """
                # Build filter_params correctly: [query_embedding, tables, ...keyword_params..., ...date_params..., query_embedding, top_k]
                table_filter_params = [query_embedding, tables] + filter_params[1:] + [query_embedding, top_k]
                filter_params = table_filter_params
            else:
                filtered_sql = normal_sql
                filter_params = normal_params
        else:
            if filter_conditions:
                filtered_sql = f"""
                    SELECT 
                        intent_id, 
                        title, 
                        description, 
                        created_at,
                        updated_at,
                        embedding <=> %s::vector AS similarity_score
                    FROM intent_chunks
                    WHERE {' AND '.join(filter_conditions)}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """
                filter_params.extend([query_embedding, top_k])
            else:
                filtered_sql = normal_sql
                filter_params = normal_params
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Normal search
                    cursor.execute(normal_sql, normal_params)
                    normal_results = cursor.fetchall()
                    
                    # Filtered search (if filters exist)
                    filtered_results = []
                    if keywords or (table_specific and tables):
                        cursor.execute(filtered_sql, filter_params)
                        filtered_results = cursor.fetchall()
                    
                    logger.info(f"Found {len(normal_results)} intents (normal), {len(filtered_results)} intents (filtered)")
                    return normal_results, filtered_results
                    
        except Exception as e:
            logger.error(f"Database error in intents search: {e}")
            return [], []
    
    def search_similar_chunks_with_analysis(self, query_embedding: List[float], 
                                          intent_ids: List[str],
                                          keywords: List[str] = None,
                                          date_range: Dict[str, str] = None,
                                          use_combined_embedding: bool = False,
                                          top_k: int = 25,
                                          table_specific: bool = False,
                                          tables: List[str] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Search chunks with dual embedding support - returns (normal_results, filtered_results, overlap_results).
        
        Args:
            query_embedding: The query embedding vector
            intent_ids: List of intent IDs to search within
            keywords: Optional keywords for filtering
            date_range: Optional date range for filtering
            use_combined_embedding: If True, use combined_embedding column; if False, use embedding column
            top_k: Number of results to return
        """
        
        if not intent_ids:
            return [], [], []
        
        # Choose embedding column based on flag
        embedding_field = "combined_embedding" if use_combined_embedding else "embedding"
        
        # Normal search without filters
        if table_specific and tables:
            normal_sql = f"""
                SELECT 
                    dc.chunk_id, 
                    dc.combined_context, 
                    dc.intent_id, 
                    dc.mapped_to_intent,
                    dc.title,
                    dc.summary,
                    '{embedding_field}' AS embedding_source,
                    dc.{embedding_field} <=> %s::vector AS similarity_score
                FROM document_chunks dc
                INNER JOIN document_files df ON dc.file_id = df.file_id
                WHERE dc.intent_id = ANY(%s) AND dc.mapped_to_intent = true AND df.table_name = ANY(%s::text[])
                ORDER BY dc.{embedding_field} <=> %s::vector
                LIMIT %s;
            """
            normal_params = [query_embedding, intent_ids, tables, query_embedding, top_k]
        else:
            normal_sql = f"""
                SELECT 
                    chunk_id, 
                    combined_context, 
                    intent_id, 
                    mapped_to_intent,
                    title,
                    summary,
                    '{embedding_field}' AS embedding_source,
                    {embedding_field} <=> %s::vector AS similarity_score
                FROM document_chunks
                WHERE intent_id = ANY(%s) AND mapped_to_intent = true
                ORDER BY {embedding_field} <=> %s::vector
                LIMIT %s;
            """
            normal_params = [query_embedding, intent_ids, query_embedding, top_k]
        
        # Filtered search - INCLUDING title and summary in both SELECT and WHERE
        if table_specific and tables:
            # Use table alias when joining tables
            filter_conditions = ["dc.intent_id = ANY(%s)", "dc.mapped_to_intent = true"]
        else:
            # Use unqualified names when not joining
            filter_conditions = ["intent_id = ANY(%s)", "mapped_to_intent = true"]
        filter_params = [query_embedding, intent_ids]
        
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                if table_specific and tables:
                    # Use table alias when joining tables
                    keyword_conditions.append("(dc.combined_context ILIKE %s OR dc.title ILIKE %s OR dc.summary ILIKE %s)")
                else:
                    # Use unqualified names when not joining
                    keyword_conditions.append("(combined_context ILIKE %s OR title ILIKE %s OR summary ILIKE %s)")
                filter_params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
            
            if keyword_conditions:
                filter_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        if table_specific and tables:
            # Add table filtering to filtered search
            if filter_conditions:
                filtered_sql = f"""
                    SELECT 
                        dc.chunk_id, 
                        dc.combined_context, 
                        dc.intent_id, 
                        dc.mapped_to_intent,
                        dc.title,
                        dc.summary,
                        '{embedding_field}' AS embedding_source,
                        dc.{embedding_field} <=> %s::vector AS similarity_score
                    FROM document_chunks dc
                    INNER JOIN document_files df ON dc.file_id = df.file_id
                    WHERE {' AND '.join(filter_conditions)} AND df.table_name = ANY(%s::text[])
                    ORDER BY dc.{embedding_field} <=> %s::vector
                    LIMIT %s;
                """
                # Build filter_params correctly: [query_embedding, intent_ids, ...keyword_params..., tables, query_embedding, top_k]
                table_filter_params = filter_params + [tables, query_embedding, top_k]
                filter_params = table_filter_params
            else:
                filtered_sql = normal_sql
                filter_params = normal_params
        else:
            if filter_conditions:
                filtered_sql = f"""
                    SELECT 
                        chunk_id, 
                        combined_context, 
                        intent_id, 
                        mapped_to_intent,
                        title,
                        summary,
                        '{embedding_field}' AS embedding_source,
                        {embedding_field} <=> %s::vector AS similarity_score
                    FROM document_chunks
                    WHERE {' AND '.join(filter_conditions)}
                    ORDER BY {embedding_field} <=> %s::vector
                    LIMIT %s;
                """
                filter_params.extend([query_embedding, top_k])
            else:
                filtered_sql = normal_sql
                filter_params = normal_params
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Normal search
                    cursor.execute(normal_sql, normal_params)
                    normal_results = cursor.fetchall()
                    
                    # Filtered search (if filters exist)
                    filtered_results = []
                    if keywords or (table_specific and tables):
                        cursor.execute(filtered_sql, filter_params)
                        filtered_results = cursor.fetchall()
                    
                    # Find overlaps
                    normal_ids = {chunk['chunk_id'] for chunk in normal_results}
                    filtered_ids = {chunk['chunk_id'] for chunk in filtered_results}
                    overlap_ids = normal_ids.intersection(filtered_ids)
                    
                    overlap_results = [chunk for chunk in normal_results if chunk['chunk_id'] in overlap_ids]
                    
                    logger.info(f"Chunks ({embedding_field}) - Normal: {len(normal_results)}, Filtered: {len(filtered_results)}, Overlap: {len(overlap_results)}")
                    return normal_results, filtered_results, overlap_results
                    
        except Exception as e:
            logger.error(f"Database error in chunks search ({embedding_field}): {e}")
            return [], [], []

# ======= Enhanced Search Pipeline with Dual Embeddings =======
class DocumentSearchPipelineV2:
    """Enhanced pipeline with dual embedding support and customizable re-ranker options."""
    
    def __init__(self, db_config: DatabaseConfig, embedding_generator: EmbeddingGenerator, 
                 query_analyzer: EnhancedQueryAnalyzer, reranker: LLMReranker):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.query_analyzer = query_analyzer
        self.reranker = reranker
        self.search_engine = EnhancedVectorSearchEngine(db_config)
    
    def _merge_chunks(self, chunks_reg: List[Dict], chunks_comb: List[Dict], limit: int) -> List[Dict]:
        """Merge regular and combined embedding chunks, prioritizing by score."""
        chunk_map = {}
        
        # Add regular chunks
        for chunk in chunks_reg:
            chunk_id = chunk['chunk_id']
            chunk_map[chunk_id] = {**chunk, 'source': 'regular'}
        
        # Add combined chunks, updating if better score
        for chunk in chunks_comb:
            chunk_id = chunk['chunk_id']
            if chunk_id in chunk_map:
                # Keep the version with better (lower) similarity score
                if chunk['similarity_score'] < chunk_map[chunk_id]['similarity_score']:
                    chunk_map[chunk_id] = {**chunk, 'source': 'combined'}
            else:
                chunk_map[chunk_id] = {**chunk, 'source': 'combined'}
        
        # Sort by similarity score and return top results
        merged_chunks = list(chunk_map.values())
        merged_chunks.sort(key=lambda x: x['similarity_score'])
        return merged_chunks[:limit]
    
    async def search(self, query: str, 
                    use_intent_reranker: bool = True,
                    use_chunk_reranker: bool = True,
                    use_dual_embeddings: bool = True,
                    intent_top_k: int = 8, 
                    chunk_top_k: int = 20,
                    table_specific: bool = False,
                    tables: List[str] = None) -> Dict[str, Any]:
        """
        Enhanced search pipeline with dual embedding support.
        
        Args:
            query: Search query
            use_intent_reranker: Enable/disable intent re-ranking
            use_chunk_reranker: Enable/disable chunk re-ranking
            use_dual_embeddings: Enable/disable dual embedding search
            intent_top_k: Number of top intents to retrieve
            chunk_top_k: Number of top chunks to retrieve
            table_specific: If True, search only in specified tables
            tables: List of table names to search in when table_specific is True
        """
        
        logger.info(f"Starting search pipeline for query: {query[:50]}...")
        
        # Step 1: Analyze query with LLM
        logger.info("Step 1: Analyzing query with LLM...")
        analysis = await self.query_analyzer.analyze_query(query)
        
        # Step 2: Generate query embedding
        logger.info("Step 2: Generating query embedding...")
        query_embedding = await self.embedding_generator.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # Step 3: Search intents
        logger.info("Step 3: Searching intents...")
        keywords = analysis.get('keywords', []) if analysis.get('filter', False) else None
        date_range = analysis.get('date_range') if analysis.get('filter', False) else None
        
        intent_normal, intent_filtered = self.search_engine.search_similar_intents(
            query_embedding, keywords, date_range, intent_top_k, table_specific, tables
        )
        
        if not intent_normal:
            return {"error": "No matching intents found"}
        
        # Step 4: Rerank intents if enabled
        intent_reranked = intent_normal
        if use_intent_reranker:
            logger.info("Step 4a: Re-ranking intents...")
            intent_reranked = await self.reranker.rerank_intents(query, intent_normal)
        
        # Use top intents for chunk search
        intent_ids = [result['intent_id'] for result in intent_reranked[:intent_top_k//2 + 1]]
        
        # Step 5: Search chunks with dual embeddings
        logger.info("Step 5: Searching chunks...")
        
        if use_dual_embeddings:
            logger.info("Step 5a: Searching with regular embeddings...")
            # Regular embedding chunks
            chunk_normal_reg, chunk_filtered_reg, chunk_overlap_reg = self.search_engine.search_similar_chunks_with_analysis(
                query_embedding, intent_ids, keywords, date_range, False, chunk_top_k, table_specific, tables  # False = regular embedding
            )
            
            logger.info("Step 5b: Searching with combined embeddings...")
            # Combined embedding chunks  
            chunk_normal_comb, chunk_filtered_comb, chunk_overlap_comb = self.search_engine.search_similar_chunks_with_analysis(
                query_embedding, intent_ids, keywords, date_range, True, chunk_top_k, table_specific, tables   # True = combined embedding
            )
            
            logger.info("Step 5c: Merging dual embedding results...")
            # Merge chunk results
            chunk_normal = self._merge_chunks(chunk_normal_reg, chunk_normal_comb, chunk_top_k)
            chunk_filtered = self._merge_chunks(chunk_filtered_reg, chunk_filtered_comb, chunk_top_k)
            chunk_overlap = self._merge_chunks(chunk_overlap_reg, chunk_overlap_comb, chunk_top_k)
            
            # Store separate results for analysis
            chunk_results_detailed = {
                "regular": {
                    "normal": chunk_normal_reg,
                    "filtered": chunk_filtered_reg,
                    "overlap": chunk_overlap_reg
                },
                "combined": {
                    "normal": chunk_normal_comb,
                    "filtered": chunk_filtered_comb,
                    "overlap": chunk_overlap_comb
                }
            }
        else:
            # Use only regular embeddings
            chunk_normal, chunk_filtered, chunk_overlap = self.search_engine.search_similar_chunks_with_analysis(
                query_embedding, intent_ids, keywords, date_range, False, chunk_top_k, table_specific, tables
            )
            chunk_results_detailed = None
        
        # Step 6: Rerank chunks if enabled
        chunks_reranked = chunk_normal
        if use_chunk_reranker and chunk_normal:
            logger.info("Step 6a: Re-ranking merged chunks...")
            chunks_reranked = await self.reranker.rerank_chunks(query, chunk_normal)
        
        result = {
            "query": query,
            "analysis": analysis,
            "reranker_config": {
                "use_intent_reranker": use_intent_reranker,
                "use_chunk_reranker": use_chunk_reranker,
                "use_dual_embeddings": use_dual_embeddings,
                "table_specific": table_specific,
                "tables": tables
            },
            "intent_results": {
                "normal": intent_normal,
                "filtered": intent_filtered,
                "reranked": intent_reranked
            },
            "chunk_results": {
                "normal": chunk_normal,
                "filtered": chunk_filtered,
                "overlap": chunk_overlap,
                "reranked": chunks_reranked
            },
            "stats": {
                "intents_kept": f"{len(intent_reranked)}/{len(intent_normal)}",
                "chunks_kept": f"{len(chunks_reranked)}/{len(chunk_normal)}"
            }
        }
        
        # Add detailed dual embedding results if available
        if chunk_results_detailed:
            result["dual_embedding_details"] = chunk_results_detailed
            result["stats"]["dual_embedding_stats"] = {
                "regular_chunks": len(chunk_results_detailed["regular"]["normal"]),
                "combined_chunks": len(chunk_results_detailed["combined"]["normal"]),
                "merged_chunks": len(chunk_normal)
            }
        
        return result

def display_search_results(results: Dict[str, Any]):
    """Display search results with dual embedding analysis."""
    
    print(f"\n{'='*100}")
    print(f"SEARCH RESULTS FOR: {results['query']}")
    print(f"{'='*100}")
    
    # Query Analysis
    analysis = results['analysis']
    print(f"\n🔍 QUERY ANALYSIS:")
    print("-" * 40)
    print(f"Keywords: {analysis.get('keywords', [])}")
    print(f"Date Range: {analysis.get('date_range', 'None')}")
    print(f"Filtering Applied: {analysis.get('filter', False)}")
    
    # Re-ranker Configuration
    reranker_config = results['reranker_config']
    print(f"\n🤖 CONFIGURATION:")
    print("-" * 40)
    print(f"Intent Re-ranker: {'✅ Enabled' if reranker_config['use_intent_reranker'] else '❌ Disabled'}")
    print(f"Chunk Re-ranker: {'✅ Enabled' if reranker_config['use_chunk_reranker'] else '❌ Disabled'}")
    print(f"Dual Embeddings: {'✅ Enabled' if reranker_config['use_dual_embeddings'] else '❌ Disabled'}")
    print(f"Table Specific: {'✅ Enabled' if reranker_config.get('table_specific', False) else '❌ Disabled'}")
    if reranker_config.get('table_specific', False) and reranker_config.get('tables'):
        print(f"Tables: {', '.join(reranker_config['tables'])}")
    
    # Statistics
    stats = results.get('stats', {})
    print(f"\n📊 STATISTICS:")
    print("-" * 40)
    print(f"Intents kept: {stats.get('intents_kept', 'N/A')}")
    print(f"Chunks kept: {stats.get('chunks_kept', 'N/A')}")
    
    # Dual embedding statistics if available
    if 'dual_embedding_stats' in stats:
        dual_stats = stats['dual_embedding_stats']
        print(f"Regular embedding chunks: {dual_stats['regular_chunks']}")
        print(f"Combined embedding chunks: {dual_stats['combined_chunks']}")
        print(f"Merged chunks: {dual_stats['merged_chunks']}")
    
    # Intent Results
    intent_results = results['intent_results']
    print(f"\n🎯 INTENT RESULTS:")
    print("-" * 50)
    print(f"Normal Results: {len(intent_results['normal'])}")
    print(f"Filtered Results: {len(intent_results['filtered'])}")
    print(f"Reranked Results: {len(intent_results['reranked'])}")
    
    print(f"\n📋 TOP INTENTS:")
    for i, result in enumerate(intent_results['reranked'][:5], 1):
        rerank_pos = result.get('rerank_position', 'N/A')
        orig_pos = result.get('original_position', 'N/A')
        print(f"{i}. [Score: {result['similarity_score']:.4f}] [Pos: {orig_pos}→{rerank_pos}]")
        print(f"   Title: {result['title']}")
        print(f"   Description: {result['description'][:100]}...")
        print()
    
    # Chunk Results
    chunk_results = results['chunk_results']
    print(f"\n📝 CHUNK RESULTS:")
    print("-" * 50)
    print(f"Normal Results: {len(chunk_results['normal'])}")
    print(f"Filtered Results: {len(chunk_results['filtered'])}")
    print(f"Overlap Results: {len(chunk_results['overlap'])}")
    print(f"Reranked Results: {len(chunk_results['reranked'])}")
    
    print(f"\n🏆 TOP CHUNKS:")
    for i, result in enumerate(chunk_results['reranked'][:5], 1):
        rerank_pos = result.get('rerank_position', 'N/A')
        orig_pos = result.get('original_position', 'N/A')
        embedding_source = result.get('source', 'N/A')
        print(f"{i}. [Score: {result['similarity_score']:.4f}] [Pos: {orig_pos}→{rerank_pos}] [Source: {embedding_source}]")
        print(f"   Intent ID: {result.get('intent_id', 'N/A')}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Summary: {result.get('summary', 'N/A')[:100]}...")
        print(f"   Content: {result.get('combined_context', '')[:100]}...")
        print()
    
    # Dual Embedding Details if available
    if 'dual_embedding_details' in results:
        dual_details = results['dual_embedding_details']
        print(f"\n🔬 DUAL EMBEDDING ANALYSIS:")
        print("-" * 60)
        
        print(f"\n📈 REGULAR EMBEDDING RESULTS:")
        print(f"   Normal: {len(dual_details['regular']['normal'])}")
        print(f"   Filtered: {len(dual_details['regular']['filtered'])}")
        print(f"   Overlap: {len(dual_details['regular']['overlap'])}")
        
        print(f"\n📊 COMBINED EMBEDDING RESULTS:")
        print(f"   Normal: {len(dual_details['combined']['normal'])}")
        print(f"   Filtered: {len(dual_details['combined']['filtered'])}")
        print(f"   Overlap: {len(dual_details['combined']['overlap'])}")
        
        # Show top results from each embedding type
        print(f"\n🔍 TOP REGULAR EMBEDDING CHUNKS:")
        for i, chunk in enumerate(dual_details['regular']['normal'][:3], 1):
            print(f"   {i}. [Score: {chunk['similarity_score']:.4f}] {chunk.get('title', 'N/A')}")
            print(f"      Content: {chunk.get('combined_context', '')[:80]}...")
        
        print(f"\n🔍 TOP COMBINED EMBEDDING CHUNKS:")
        for i, chunk in enumerate(dual_details['combined']['normal'][:3], 1):
            print(f"   {i}. [Score: {chunk['similarity_score']:.4f}] {chunk.get('title', 'N/A')}")
            print(f"      Content: {chunk.get('combined_context', '')[:80]}...")
        
        # Show overlap analysis
        regular_ids = {chunk['chunk_id'] for chunk in dual_details['regular']['normal']}
        combined_ids = {chunk['chunk_id'] for chunk in dual_details['combined']['normal']}
        overlap_ids = regular_ids.intersection(combined_ids)
        
        print(f"\n📊 EMBEDDING OVERLAP ANALYSIS:")
        print(f"   Regular unique chunks: {len(regular_ids - combined_ids)}")
        print(f"   Combined unique chunks: {len(combined_ids - regular_ids)}")
        print(f"   Overlapping chunks: {len(overlap_ids)}")
        print(f"   Total unique chunks: {len(regular_ids.union(combined_ids))}")
    
    print(f"\n{'='*100}")
    print("SEARCH COMPLETED")
    print(f"{'='*100}")


# ======= Main Execution Function =======
async def main():
    """Main function to demonstrate the enhanced search pipeline."""
    
    # Configuration
    API_KEY = os.getenv("google_api_key")
    if not API_KEY:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return
    
    db_config = DatabaseConfig()
    
    # Initialize components
    print("🔧 Initializing search components...")
    embedding_generator = EmbeddingGenerator(API_KEY)
    query_analyzer = EnhancedQueryAnalyzer(API_KEY)
    reranker = LLMReranker(API_KEY)
    search_pipeline = DocumentSearchPipelineV2(db_config, embedding_generator, query_analyzer, reranker)
    
    # Example queries to test
    test_queries = [
        "how is the chariman of ific bank?"
    ]
    
    print("🚀 Starting search pipeline tests...")
    print(f"{'='*100}")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 TEST {i}/{len(test_queries)}: {query}")
        print("-" * 80)
        
        try:
            # Test with different configurations
            configurations = [
                {"use_intent_reranker": True, "use_chunk_reranker": True, "use_dual_embeddings": False},
                # {"use_intent_reranker": False, "use_chunk_reranker": False, "use_dual_embeddings": False},
                # {"use_intent_reranker": True, "use_chunk_reranker": True, "use_dual_embeddings": False},
            ]
            
            for config_idx, config in enumerate(configurations, 1):
                print(f"\n📋 Configuration {config_idx}: {config}")
                
                results = await search_pipeline.search(
                    query=query,
                    **config
                )
                
                if "error" in results:
                    print(f"❌ Error: {results['error']}")
                    continue
                
                display_search_results(results)
                
                # Add a separator between configurations
                if config_idx < len(configurations):
                    print(f"\n{'-'*50}")
                    
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")
            logger.error(f"Error in main execution: {e}")
    
    print(f"\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())