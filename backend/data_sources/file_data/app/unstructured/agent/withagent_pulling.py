import os
import asyncio
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
import google.generativeai as genai
import numpy as np
import json
from datetime import datetime

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

# ======= Enhanced Query Analyzer =======
class EnhancedQueryAnalyzer:
    """Analyzes queries using LLM to extract keywords and dates."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-05-20"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract keywords, dates, and filtering decision."""
        
        prompt = f"""
You are a smart document analysis assistant.
Given the following query, perform these tasks:

1. Extract ONLY significant keywords that would be useful for filtering documents. 
   - Include: names, organizations, places, important concepts, unique terms, technical terms
   - Exclude: common words, articles, prepositions, generic terms
   - Only include keywords that would meaningfully narrow down search results

2. Detect any date mentions. If dates are present, identify the earliest and latest dates in format `YYYY-MM-DD`. If no dates, return null.

3. Determine if keyword or date filtering will be helpful based on specificity and context.

Return STRICTLY in this JSON format:
{{
  "keywords": ["keyword1", "keyword2", ...],
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
  "filter": true/false
}}

Query: {query}
"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content, 
                prompt
            )
            
            response_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text
            
            analysis = json.loads(json_text)
            logger.info(f"Query analysis: {analysis}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return {
                "keywords": [],
                "date_range": None,
                "filter": False
            }

# ======= LLM Re-ranker =======
class LLMReranker:
    """Re-ranks search results using LLM to remove irrelevant items and reorder."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite-preview-06-17"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    async def rerank_subintents(self, query: str, subintents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank sub-intents based on relevance to query."""
        
        if not subintents:
            return []
        
        # Prepare data for LLM
        subintent_data = []
        for i, sub in enumerate(subintents):
            subintent_data.append({
                "index": i,
                "title": sub.get('title', ''),
                "description": sub.get('description', ''),
            })
        
        prompt = f"""
You are a document relevance expert. Given a user query and a list of sub-intents, your task is to:

1. Remove sub-intents that are NOT relevant to the query
2. Reorder the remaining sub-intents by relevance (most relevant first)
3. Keep only sub-intents that would genuinely help answer the query

Query: "{query}"

Sub-intents:
{json.dumps(subintent_data, indent=2)}

Return ONLY a JSON array of indices (original positions) in order of relevance:
[0, 2, 1] (example - keep only relevant ones, reordered by relevance)

If no sub-intents are relevant, return: []
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
                
                # Return reordered subintents
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(subintents):
                        reordered_item = subintents[idx].copy()
                        reordered_item['rerank_position'] = len(reranked) + 1
                        reordered_item['original_position'] = idx + 1
                        reranked.append(reordered_item)
                
                logger.info(f"Reranked {len(reranked)}/{len(subintents)} sub-intents")
                return reranked
            
        except Exception as e:
            logger.error(f"Failed to rerank sub-intents: {e}")
        
        return subintents  # Return original if reranking fails
    
    async def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query."""
        
        if not documents:
            return []
        
        # Prepare data for LLM
        doc_data = []
        for i, doc in enumerate(documents):
            doc_data.append({
                "index": i,
                "file_name": doc.get('file_name', ''),
                "title": doc.get('title', ''),
                "summary": doc.get('full_summary', '') + "..." if doc.get('full_summary') else '',
                "keywords": doc.get('keywords', [])
            })
        
        prompt = f"""
You are a document relevance expert. Given a user query and a list of documents, your task is to:

1. Remove documents that are NOT relevant to the query
2. Reorder the remaining documents by relevance (most relevant first)
3. Keep only documents that would genuinely help answer the query

Query: "{query}"

Documents:
{json.dumps(doc_data, indent=2)}

Return ONLY a JSON array of indices (original positions) in order of relevance:
[0, 2, 1] (example - keep only relevant ones, reordered by relevance)

If no documents are relevant, return: []
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
                
                # Return reordered documents
                reranked = []
                for idx in indices:
                    if 0 <= idx < len(documents):
                        reordered_item = documents[idx].copy()
                        reordered_item['rerank_position'] = len(reranked) + 1
                        reordered_item['original_position'] = idx + 1
                        reranked.append(reordered_item)
                
                logger.info(f"Reranked {len(reranked)}/{len(documents)} documents")
                return reranked
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
        
        return documents  # Return original if reranking fails
    
    async def rerank_chunks(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank chunks based on relevance to query."""
        
        if not chunks:
            return []
        
        # Prepare data for LLM - limit chunk text for token efficiency
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "index": i,
                "summary": chunk.get('summary', ''),
                "title": chunk.get('title', ''),
                "score": chunk.get('similarity_score', 0),
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

# ======= Enhanced Vector Search Functions =======
class EnhancedVectorSearchEngine:
    """Enhanced vector search with keyword and date filtering capabilities."""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
    
    def search_similar_subintents(self, query_embedding: List[float], top_k: int = 10) -> Tuple[List[Dict], List[Dict]]:
        """Search for similar sub-intents - returns (normal_results, filtered_results)."""
        
        # Normal vector search
        normal_sql = """
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
                    # Normal search
                    cursor.execute(normal_sql, (query_embedding, query_embedding, top_k))
                    normal_results = cursor.fetchall()
                    
                    logger.info(f"Found {len(normal_results)} sub-intents (normal search)")
                    return normal_results, []  # No filtering for sub-intents currently
                    
        except Exception as e:
            logger.error(f"Database error in sub-intents search: {e}")
            return [], []
    
    def search_similar_documents_with_analysis(self, query_embedding: List[float], 
                                             sub_intent_ids: List[int], 
                                             keywords: List[str] = None,
                                             date_range: Dict[str, str] = None,
                                             top_k: int = 15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Search documents - returns (normal_results, filtered_results, overlap_results)."""
        
        # Normal search without filters
        normal_sql = """
            SELECT 
                file_id, file_name, file_type, title, full_summary, intent, sub_intent,
                sub_intent_id, keywords, date_range,
                title_summary_embedding <=> %s::vector AS similarity_score
            FROM document_files
            WHERE sub_intent_id = ANY(%s)
            ORDER BY title_summary_embedding <=> %s::vector
            LIMIT %s;
        """
        
        # Filtered search with keywords/dates
        filter_conditions = ["sub_intent_id = ANY(%s)"]
        filter_params = [query_embedding, sub_intent_ids]
        
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("(%s = ANY(keywords) OR title ILIKE %s OR full_summary ILIKE %s)")
                filter_params.extend([keyword, f'%{keyword}%', f'%{keyword}%'])
            
            if keyword_conditions:
                filter_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        if date_range and date_range.get('start') and date_range.get('end'):
            filter_conditions.append("""
                (date_range IS NOT NULL AND 
                 (date_range->>'start')::date <= %s::date AND 
                 (date_range->>'end')::date >= %s::date)
            """)
            filter_params.extend([date_range['end'], date_range['start']])
        
        filtered_sql = f"""
            SELECT 
                file_id, file_name, file_type, title, full_summary, intent, sub_intent,
                sub_intent_id, keywords, date_range,
                title_summary_embedding <=> %s::vector AS similarity_score
            FROM document_files
            WHERE {' AND '.join(filter_conditions)}
            ORDER BY title_summary_embedding <=> %s::vector
            LIMIT %s;
        """
        
        filter_params.extend([query_embedding, top_k])
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Normal search
                    cursor.execute(normal_sql, (query_embedding, sub_intent_ids, query_embedding, top_k))
                    normal_results = cursor.fetchall()
                    
                    # Filtered search (if filters exist)
                    filtered_results = []
                    if keywords or date_range:
                        cursor.execute(filtered_sql, filter_params)
                        filtered_results = cursor.fetchall()
                    
                    # Find overlaps
                    normal_ids = {doc['file_id'] for doc in normal_results}
                    filtered_ids = {doc['file_id'] for doc in filtered_results}
                    overlap_ids = normal_ids.intersection(filtered_ids)
                    
                    overlap_results = [doc for doc in normal_results if doc['file_id'] in overlap_ids]
                    
                    logger.info(f"Documents - Normal: {len(normal_results)}, Filtered: {len(filtered_results)}, Overlap: {len(overlap_results)}")
                    return normal_results, filtered_results, overlap_results
                    
        except Exception as e:
            logger.error(f"Database error in documents search: {e}")
            return [], [], []
    
    def search_similar_chunks_with_analysis(self, query_embedding: List[float], 
                                          file_ids: List[str],
                                          keywords: List[str] = None,
                                          date_range: Dict[str, str] = None,
                                          use_combined_embedding: bool = False,
                                          top_k: int = 25) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Search chunks - returns (normal_results, filtered_results, overlap_results)."""
        
        if not file_ids:
            return [], [], []
        
        embedding_field = "combined_embedding" if use_combined_embedding else "embedding"
        
        # Normal search without filters
        normal_sql = f"""
            SELECT 
                chunk_id, file_id, chunk_text, summary, title, keywords, chunk_order,
                combined_context, date_range,
                {embedding_field} <=> %s::vector AS similarity_score
            FROM document_chunks
            WHERE file_id = ANY(%s)
            ORDER BY {embedding_field} <=> %s::vector
            LIMIT %s;
        """
        
        # Filtered search
        filter_conditions = ["file_id = ANY(%s)"]
        filter_params = [query_embedding, file_ids]
        
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("(%s = ANY(keywords) OR chunk_text ILIKE %s OR summary ILIKE %s)")
                filter_params.extend([keyword, f'%{keyword}%', f'%{keyword}%'])
            
            if keyword_conditions:
                filter_conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        if date_range and date_range.get('start') and date_range.get('end'):
            filter_conditions.append("""
                (date_range IS NOT NULL AND 
                 (date_range->>'start')::date <= %s::date AND 
                 (date_range->>'end')::date >= %s::date)
            """)
            filter_params.extend([date_range['end'], date_range['start']])
        
        filtered_sql = f"""
            SELECT 
                chunk_id, file_id, chunk_text, summary, title, keywords, chunk_order,
                combined_context, date_range,
                {embedding_field} <=> %s::vector AS similarity_score
            FROM document_chunks
            WHERE {' AND '.join(filter_conditions)}
            ORDER BY {embedding_field} <=> %s::vector
            LIMIT %s;
        """
        
        filter_params.extend([query_embedding, top_k])
        
        try:
            with psycopg2.connect(self.db_config.get_connection_string()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Normal search
                    cursor.execute(normal_sql, (query_embedding, file_ids, query_embedding, top_k))
                    normal_results = cursor.fetchall()
                    
                    # Filtered search (if filters exist)
                    filtered_results = []
                    if keywords or date_range:
                        cursor.execute(filtered_sql, filter_params)
                        filtered_results = cursor.fetchall()
                    
                    # Find overlaps
                    normal_ids = {chunk['chunk_id'] for chunk in normal_results}
                    filtered_ids = {chunk['chunk_id'] for chunk in filtered_results}
                    overlap_ids = normal_ids.intersection(filtered_ids)
                    
                    overlap_results = [chunk for chunk in normal_results if chunk['chunk_id'] in overlap_ids]
                    
                    embedding_type = "combined" if use_combined_embedding else "regular"
                    logger.info(f"Chunks ({embedding_type}) - Normal: {len(normal_results)}, Filtered: {len(filtered_results)}, Overlap: {len(overlap_results)}")
                    return normal_results, filtered_results, overlap_results
                    
        except Exception as e:
            logger.error(f"Database error in chunks search: {e}")
            return [], [], []

# ======= Enhanced Search Pipeline =======
class UltimateDocumentSearchPipeline:
    """Ultimate pipeline with all features: filtering, overlap analysis, and LLM re-ranking."""
    
    def __init__(self, db_config: DatabaseConfig, embedding_generator: EmbeddingGenerator, 
                 query_analyzer: EnhancedQueryAnalyzer, reranker: LLMReranker):
        self.db_config = db_config
        self.embedding_generator = embedding_generator
        self.query_analyzer = query_analyzer
        self.reranker = reranker
        self.search_engine = EnhancedVectorSearchEngine(db_config)
    
    async def search(self, query: str, 
                    use_reranker: bool = True,
                    subintent_top_k: int = 8, 
                    document_top_k: int = 12, 
                    chunk_top_k: int = 20) -> Dict[str, Any]:
        """
        Ultimate search pipeline with all features.
        """
        
        logger.info(f"Starting ultimate search pipeline for query: {query[:50]}...")
        
        # Step 1: Analyze query with LLM
        logger.info("Step 1: Analyzing query with LLM...")
        analysis = await self.query_analyzer.analyze_query(query)
        
        # Step 2: Generate query embedding
        logger.info("Step 2: Generating query embedding...")
        query_embedding = await self.embedding_generator.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # Step 3: Search sub-intents
        logger.info("Step 3: Searching sub-intents...")
        subintent_normal, subintent_filtered = self.search_engine.search_similar_subintents(query_embedding, subintent_top_k)
        
        if not subintent_normal:
            return {"error": "No matching sub-intents found"}
        
        # Step 4: Rerank sub-intents if enabled
        subintent_reranked = subintent_normal
        if use_reranker:
            logger.info("Step 4a: Re-ranking sub-intents...")
            subintent_reranked = await self.reranker.rerank_subintents(query, subintent_normal)
        
        # Use top sub-intents for further search
        sub_intent_ids = [result['sub_intent_id'] for result in subintent_reranked[:subintent_top_k//2 + 1]]
        
        # Step 5: Search documents
        logger.info("Step 5: Searching documents...")
        keywords = analysis.get('keywords', []) if analysis.get('filter', False) else None
        date_range = analysis.get('date_range') if analysis.get('filter', False) else None
        
        doc_normal, doc_filtered, doc_overlap = self.search_engine.search_similar_documents_with_analysis(
            query_embedding, sub_intent_ids, keywords, date_range, document_top_k
        )
        
        # Step 6: Rerank documents if enabled
        doc_reranked = doc_normal
        if use_reranker and doc_normal:
            logger.info("Step 6a: Re-ranking documents...")
            doc_reranked = await self.reranker.rerank_documents(query, doc_normal)
        
        if not doc_reranked:
            return {
                "query": query,
                "analysis": analysis,
                "use_reranker": use_reranker,
                "subintent_results": {
                    "normal": subintent_normal,
                    "reranked": subintent_reranked
                },
                "document_results": {
                    "normal": doc_normal,
                    "filtered": doc_filtered,
                    "overlap": doc_overlap,
                    "reranked": doc_reranked
                },
                "chunk_results": {"normal": [], "filtered": [], "overlap": [], "reranked": []},
                "message": "No matching documents found"
            }
        
        # Use top documents for chunk search
        file_ids = [result['file_id'] for result in doc_reranked[:document_top_k//2 + 2]]
        
        # Step 7: Search chunks (both regular and combined embeddings)
        logger.info("Step 7: Searching chunks...")
        
        # Regular embedding chunks
        chunk_normal_reg, chunk_filtered_reg, chunk_overlap_reg = self.search_engine.search_similar_chunks_with_analysis(
            query_embedding, file_ids, keywords, date_range, False, chunk_top_k
        )
        
        # Combined embedding chunks
        chunk_normal_comb, chunk_filtered_comb, chunk_overlap_comb = self.search_engine.search_similar_chunks_with_analysis(
            query_embedding, file_ids, keywords, date_range, True, chunk_top_k
        )
        
        # Merge chunk results
        all_chunks_normal = self._merge_chunks(chunk_normal_reg, chunk_normal_comb, chunk_top_k)
        all_chunks_filtered = self._merge_chunks(chunk_filtered_reg, chunk_filtered_comb, chunk_top_k)
        all_chunks_overlap = self._merge_chunks(chunk_overlap_reg, chunk_overlap_comb, chunk_top_k)
        
        # Step 8: Rerank chunks if enabled
        chunks_reranked = all_chunks_normal
        if use_reranker and all_chunks_normal:
            logger.info("Step 8a: Re-ranking chunks...")
            chunks_reranked = await self.reranker.rerank_chunks(query, all_chunks_normal)
        
        return {
            "query": query,
            "analysis": analysis,
            "use_reranker": use_reranker,
            "subintent_results": {
                "normal": subintent_normal,
                "filtered": subintent_filtered,
                "overlap": [],  # No overlap for sub-intents
                "reranked": subintent_reranked
            },
            "document_results": {
                "normal": doc_normal,
                "filtered": doc_filtered,
                "overlap": doc_overlap,
                "reranked": doc_reranked
            },
            "chunk_results": {
                "normal": all_chunks_normal,
                "filtered": all_chunks_filtered,
                "overlap": all_chunks_overlap,
                "reranked": chunks_reranked
            },
            "stats": {
                "subintents_kept": f"{len(subintent_reranked)}/{len(subintent_normal)}",
                "documents_kept": f"{len(doc_reranked)}/{len(doc_normal)}",
                "chunks_kept": f"{len(chunks_reranked)}/{len(all_chunks_normal)}"
            }
        }
    
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
                if chunk['similarity_score'] < chunk_map[chunk_id]['similarity_score']:
                    chunk_map[chunk_id] = {**chunk, 'source': 'combined'}
        
        # Sort by similarity score and return top results
        merged_chunks = list(chunk_map.values())
        merged_chunks.sort(key=lambda x: x['similarity_score'])
        return merged_chunks[:limit]

# ======= Enhanced Display Functions =======
def display_ultimate_results(results: Dict[str, Any]):
    """Display ultimate search results with all analysis."""
    
    print(f"\n{'='*100}")
    print(f"ULTIMATE SEARCH RESULTS FOR: {results['query']}")
    print(f"{'='*100}")
    
    # Query Analysis
    analysis = results['analysis']
    print(f"\n🔍 QUERY ANALYSIS:")
    print("-" * 40)
    print(f"Keywords: {analysis.get('keywords', [])}")
    print(f"Date Range: {analysis.get('date_range', 'None')}")
    print(f"Filtering Applied: {analysis.get('filter', False)}")
    print(f"Re-ranker Used: {results.get('use_reranker', False)}")
    
    # Statistics
    stats = results.get('stats', {})
    print(f"\n📊 RE-RANKING STATISTICS:")
    print("-" * 40)
    print(f"Sub-intents kept: {stats.get('subintents_kept', 'N/A')}")
    print(f"Documents kept: {stats.get('documents_kept', 'N/A')}")
    print(f"Chunks kept: {stats.get('chunks_kept', 'N/A')}")
    
    # Sub-intents Analysis
    subintent_results = results['subintent_results']
    print(f"\n🎯 SUB-INTENTS ANALYSIS:")
    print("-" * 50)
    print(f"Normal Results: {len(subintent_results['normal'])}")
    print(f"Reranked Results: {len(subintent_results['reranked'])}")
    
    print(f"\n📋 TOP RERANKED SUB-INTENTS:")
    for i, result in enumerate(subintent_results['reranked'][:3], 1):
        rerank_pos = result.get('rerank_position', 'N/A')
        orig_pos = result.get('original_position', 'N/A')
        print(f"{i}. [Score: {result['similarity_score']:.4f}] [Rerank: {orig_pos}→{rerank_pos}] {result['title']}")
        print(f"   Description: {result['description'][:100]}...")
    
    # Documents Analysis
    doc_results = results['document_results']
    print(f"\n📄 DOCUMENTS ANALYSIS:")
    print("-" * 50)
    print(f"Normal Results: {len(doc_results['normal'])}")
    print(f"Filtered Results: {len(doc_results['filtered'])}")
    print(f"Overlap Results: {len(doc_results['overlap'])}")
    print(f"Reranked Results: {len(doc_results['reranked'])}")
    
    print(f"\n🔝 TOP RERANKED DOCUMENTS:")
    for i, result in enumerate(doc_results['reranked'][:3], 1):
        rerank_pos = result.get('rerank_position', 'N/A')
        orig_pos = result.get('original_position', 'N/A')
        print(f"{i}. [Score: {result['similarity_score']:.4f}] [Rerank: {orig_pos}→{rerank_pos}] {result['file_name']}")
        print(f"   Title: {result['title']}")
        print(f"   Keywords: {result.get('keywords', [])[:5]}")  # Show first 5 keywords
    
    # Chunks Analysis
    chunk_results = results['chunk_results']
    print(f"\n📝 CHUNKS ANALYSIS:")
    print("-" * 50)
    print(f"Normal Results: {len(chunk_results['normal'])}")
    print(f"Filtered Results: {len(chunk_results['filtered'])}")
    print(f"Overlap Results: {len(chunk_results['overlap'])}")
    print(f"Reranked Results: {len(chunk_results['reranked'])}")
    
    print(f"\n🏆 TOP RERANKED CHUNKS:")
    for i, result in enumerate(chunk_results['reranked'][:5], 1):
        rerank_pos = result.get('rerank_position', 'N/A')
        orig_pos = result.get('original_position', 'N/A')
        source_indicator = {
            'regular': '🔵',
            'combined': '🟡', 
            'regular_better': '🟢',
            'combined_better': '🟠'
        }.get(result.get('source', ''), '⚫')
        
        print(f"{i}. {source_indicator} [Score: {result['similarity_score']:.4f}] [Rerank: {orig_pos}→{rerank_pos}]")
        print(f"   Chunk {result.get('chunk_order', 'N/A')} from {result.get('file_id', 'Unknown')}")
        print(f"   Source: {result.get('source', 'unknown')}")
        print(f"   Text: {result.get('chunk_text', '')[:120]}...")
        if result.get('keywords'):
            print(f"   Keywords: {result['keywords'][:3]}")  # Show first 3 keywords
        print()

def display_comparison_analysis(results: Dict[str, Any]):
    """Display detailed comparison between normal, filtered, overlap, and reranked results."""
    
    print(f"\n{'='*100}")
    print(f"DETAILED COMPARISON ANALYSIS")
    print(f"{'='*100}")
    
    # Sub-intents Comparison
    subintent_results = results['subintent_results']
    print(f"\n🎯 SUB-INTENTS COMPARISON:")
    print("-" * 60)
    
    normal_sub = subintent_results['normal']
    reranked_sub = subintent_results['reranked']
    
    print(f"Original Order vs Reranked Order:")
    for i in range(min(5, len(normal_sub))):
        normal_item = normal_sub[i]
        
        # Find this item in reranked results
        reranked_pos = "REMOVED"
        for j, reranked_item in enumerate(reranked_sub):
            if reranked_item['sub_intent_id'] == normal_item['sub_intent_id']:
                reranked_pos = j + 1
                break
        
        print(f"  Position {i+1} → {reranked_pos}: {normal_item['title'][:50]}...")
    
    # Documents Comparison
    doc_results = results['document_results']
    print(f"\n📄 DOCUMENTS COMPARISON:")
    print("-" * 60)
    
    print(f"Normal ({len(doc_results['normal'])}) vs Filtered ({len(doc_results['filtered'])}) vs Reranked ({len(doc_results['reranked'])}):")
    
    # Show normal vs reranked comparison
    normal_docs = doc_results['normal']
    reranked_docs = doc_results['reranked']
    
    for i in range(min(5, len(normal_docs))):
        normal_item = normal_docs[i]
        
        # Find this item in reranked results
        reranked_pos = "REMOVED"
        for j, reranked_item in enumerate(reranked_docs):
            if reranked_item['file_id'] == normal_item['file_id']:
                reranked_pos = j + 1
                break
        
        print(f"  Position {i+1} → {reranked_pos}: {normal_item['file_name']}")
    
    # Overlap analysis
    if doc_results['overlap']:
        print(f"\n🔄 OVERLAP DOCUMENTS ({len(doc_results['overlap'])}):")
        for doc in doc_results['overlap'][:3]:
            print(f"  - {doc['file_name']} (Score: {doc['similarity_score']:.4f})")
    
    # Chunks Comparison
    chunk_results = results['chunk_results']
    print(f"\n📝 CHUNKS COMPARISON:")
    print("-" * 60)
    
    print(f"Normal ({len(chunk_results['normal'])}) vs Filtered ({len(chunk_results['filtered'])}) vs Reranked ({len(chunk_results['reranked'])}):")
    
    # Source distribution in reranked chunks
    source_counts = {}
    for chunk in chunk_results['reranked']:
        source = chunk.get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nSource Distribution in Reranked Results:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    # Show top reranked chunk changes
    normal_chunks = chunk_results['normal']
    reranked_chunks = chunk_results['reranked']
    
    print(f"\nTop 5 Chunk Position Changes:")
    for i in range(min(5, len(normal_chunks))):
        normal_item = normal_chunks[i]
        
        # Find this item in reranked results
        reranked_pos = "REMOVED"
        for j, reranked_item in enumerate(reranked_chunks):
            if reranked_item['chunk_id'] == normal_item['chunk_id']:
                reranked_pos = j + 1
                break
        
        print(f"  Position {i+1} → {reranked_pos}: Chunk {normal_item.get('chunk_order', 'N/A')} from {normal_item.get('file_id', 'Unknown')}")

# ======= Main Runner =======
async def main():
    """Main function to run the ultimate search pipeline."""
    
    # Configuration
    db_config = DatabaseConfig()
    api_key = os.getenv("google_api_key") or "your_google_api_key_here"
    
    if api_key == "your_google_api_key_here":
        logger.error("Please set your Google API key in the GOOGLE_API_KEY environment variable")
        return
    
    # Test queries
    test_queries = [
        "Bill of Quantities and Prices for Furnishing 2018?",
        "What are the procurement procedures for construction materials?",
        "Show me documents about project management from 2019-2020"
    ]
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(api_key)
    query_analyzer = EnhancedQueryAnalyzer(api_key)
    reranker = LLMReranker(api_key)
    
    search_pipeline = UltimateDocumentSearchPipeline(
        db_config, embedding_generator, query_analyzer, reranker
    )
    
    # Run searches with different configurations
    configurations = [
        {"use_reranker": False, "label": "WITHOUT RE-RANKER"},
        {"use_reranker": True, "label": "WITH RE-RANKER"}
    ]
    
    for query in test_queries:
        print(f"\n{'='*120}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*120}")
        
        for config in configurations:
            try:
                print(f"\n{'-'*60}")
                print(f"RUNNING {config['label']}")
                print(f"{'-'*60}")
                
                results = await search_pipeline.search(
                    query=query,
                    use_reranker=config['use_reranker'],
                    subintent_top_k=6,
                    document_top_k=8,
                    chunk_top_k=15
                )
                
                if "error" in results:
                    print(f"❌ Error: {results['error']}")
                    continue
                
                display_ultimate_results(results)
                

                
            except Exception as e:
                logger.error(f"Error processing query '{query}' with config {config}: {e}")
                continue
            


async def data_pull(query, use_reranker=True, subintent_top_k=6, document_top_k=8, chunk_top_k=15):
    """
    Main function to run the ultimate search pipeline with configurable parameters.
    
    Args:
        query (str): The search query to process
        use_reranker (bool): Whether to use the reranker in the search pipeline
        subintent_top_k (int): Number of top subintents to consider
        document_top_k (int): Number of top documents to retrieve
        chunk_top_k (int): Number of top chunks to retrieve
    
    Returns:
        dict: Search results from the pipeline
    """
    # Configuration
    db_config = DatabaseConfig()
    api_key = os.getenv("google_api_key") or "your_google_api_key_here"
    
    if api_key == "your_google_api_key_here":
        logger.error("Please set your Google API key in the GOOGLE_API_KEY environment variable")
        return {"error": "Google API key not configured"}
    
    try:
        # Initialize components
        embedding_generator = EmbeddingGenerator(api_key)
        query_analyzer = EnhancedQueryAnalyzer(api_key)
        reranker = LLMReranker(api_key)
        search_pipeline = UltimateDocumentSearchPipeline(
            db_config, embedding_generator, query_analyzer, reranker
        )
        
        print(f"\n{'='*120}")
        print(f"PROCESSING QUERY: {query}")
        print(f"{'='*120}")
        
        # Run search with provided configuration
        label = "WITH RE-RANKER" if use_reranker else "WITHOUT RE-RANKER"
        print(f"\n{'-'*60}")
        print(f"RUNNING {label}")
        print(f"{'-'*60}")
        
        results = await search_pipeline.search(
            query=query,
            use_reranker=use_reranker,
            subintent_top_k=subintent_top_k,
            document_top_k=document_top_k,
            chunk_top_k=chunk_top_k
        )
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return results
        
        display_ultimate_results(results)
        return results
        
    except Exception as e:
        error_msg = f"Error processing query '{query}': {e}"
        logger.error(error_msg)
        return {"error": error_msg}
if __name__ == "__main__":
    asyncio.run(data_pull("Bill of Quantities and Prices for Furnishing 2018?"))
    # Uncomment the line