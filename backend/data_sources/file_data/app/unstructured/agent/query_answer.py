import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

# Import the data_pull function from the search pipeline
from data_sources.file_data.app.unstructured.agent.withagent_pulling import data_pull

# Load environment variables
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

google_gemini_name_light = os.getenv("google_gemini_name_light", "gemini-1.5-pro")
if google_gemini_name_light is None:
    print("Warning: 'google_gemini_name_light' not found in environment variables, using default 'gemini-1.5-pro'.")

thinking_model = os.getenv("thinking_model", "deepseek-r1-distill-llama-70b")
if thinking_model is None:
    print("Warning: 'groq_thinking_model' not found in environment variables, using default 'deepseek-r1-distill-llama-70b'.")


def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)


class DocumentQASystem:
    """
    Document Question Answering System that uses the search pipeline to find relevant chunks
    and then uses LLM to generate answers based on the retrieved context.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None, temperature: float = 0.1):
        """
        Initialize the QA System with LLM configuration.
        
        Args:
            api_key (str): Google API key for the LLM
            model_name (str): Name of the Gemini model to use
            temperature (float): Temperature setting for the LLM
        """
        self.api_key = api_key or gemini_apikey
        self.model_name = model_name or gemini_model_name
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("Google API key is required. Please set 'google_api_key' environment variable.")
        
        # Initialize the LLM
        self.llm = initialize_llm_gemini(
            api_key=self.api_key,
            temperature=self.temperature,
            model=self.model_name
        )
        
        logger.info(f"Initialized DocumentQASystem with model: {self.model_name}")
    
    def extract_chunks_text(self, search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relevant chunk information from search results.
        
        Args:
            search_results (Dict): Results from the data_pull function
            
        Returns:
            List[Dict]: List of chunk information with text, metadata, etc.
        """
        if "error" in search_results:
            logger.error(f"Search results contain error: {search_results['error']}")
            return []
        
        chunk_results = search_results.get('chunk_results', {})
        reranked_chunks = chunk_results.get('reranked', [])
        
        if not reranked_chunks:
            logger.warning("No reranked chunks found in search results")
            return []
        
        # Extract relevant information from chunks
        processed_chunks = []
        for i, chunk in enumerate(reranked_chunks):
            chunk_info = {
                'rank': i + 1,
                'chunk_id': chunk.get('chunk_id'),
                'file_id': chunk.get('file_id'),
                'chunk_text': chunk.get('chunk_text', ''),
                'summary': chunk.get('summary', ''),
                'title': chunk.get('title', ''),
                'similarity_score': chunk.get('similarity_score', 0),
                'keywords': chunk.get('keywords', []),
                'chunk_order': chunk.get('chunk_order', 'N/A'),
                'source': chunk.get('source', 'unknown')
            }
            processed_chunks.append(chunk_info)
        
        logger.info(f"Extracted {len(processed_chunks)} chunks from search results")
        return processed_chunks
    
    def create_context_from_chunks(self, chunks: List[Dict[str, Any]], max_chunks: int = 10) -> str:
        """
        Create a context string from the most relevant chunks.
        
        Args:
            chunks (List[Dict]): List of chunk information
            max_chunks (int): Maximum number of chunks to include in context
            
        Returns:
            str: Formatted context string for the LLM
        """
        if not chunks:
            return ""
        
        context_parts = []
        
        # Use the top chunks (already ranked by relevance)
        selected_chunks = chunks[:max_chunks]
        
        for chunk in selected_chunks:
            chunk_text = chunk.get('chunk_text', '').strip()
            if chunk_text:
                # Add chunk metadata for better context
                file_id = chunk.get('file_id', 'Unknown')
                chunk_order = chunk.get('chunk_order', 'N/A')
                similarity_score = chunk.get('similarity_score', 0)
                
                chunk_header = f"[Document: {file_id} | Chunk: {chunk_order} | Relevance Score: {similarity_score:.4f}]"
                context_parts.append(f"{chunk_header}\n{chunk_text}")
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        logger.info(f"Created context from {len(selected_chunks)} chunks")
        return context
    
    def create_qa_prompt(self, query: str, context: str) -> List:
        """
        Create a structured prompt for the LLM to answer questions based on context.
        
        Args:
            query (str): User's question
            context (str): Context from retrieved chunks
            
        Returns:
            List: List of messages for the LLM
        """
        system_prompt = """You are an expert document analysis assistant. Your task is to answer questions based ONLY on the provided document context.

IMPORTANT INSTRUCTIONS:
1. Answer ONLY using information from the provided document chunks
2. If the information needed to answer the question is not available in the chunks, respond with: "Not enough information in the chunks to answer this question."
3. Be specific and cite relevant details from the chunks when possible
4. If you find relevant information, provide a comprehensive answer
5. Do not make assumptions or add information not present in the chunks
6. If multiple chunks contain relevant information, synthesize them into a coherent answer

Response format:
- If sufficient information is available: Provide a detailed answer based on the chunks
- If insufficient information: "Not enough information in the chunks to answer this question."
"""
        
        user_prompt = f"""Based on the following document chunks, please answer this question:

QUESTION: {query}

DOCUMENT CHUNKS:
{context}

Please provide your answer based solely on the information in these chunks."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        return messages
    
    async def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM based on the query and context.
        
        Args:
            query (str): User's question
            context (str): Context from retrieved chunks
            
        Returns:
            str: Generated answer from the LLM
        """
        try:
            messages = self.create_qa_prompt(query, context)
            
            logger.info("Generating answer using LLM...")
            response = await asyncio.to_thread(self.llm.invoke, messages)
            
            answer = response.content.strip()
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def answer_question(self, 
                            query: str, 
                            use_reranker: bool = True,
                            subintent_top_k: int = 6,
                            document_top_k: int = 8,
                            chunk_top_k: int = 15,
                            max_context_chunks: int = 10) -> Dict[str, Any]:
        """
        Main method to answer a question using the document search and LLM pipeline.
        
        Args:
            query (str): User's question
            use_reranker (bool): Whether to use reranker in search pipeline
            subintent_top_k (int): Number of top subintents to consider
            document_top_k (int): Number of top documents to retrieve
            chunk_top_k (int): Number of top chunks to retrieve
            max_context_chunks (int): Maximum chunks to include in LLM context
            
        Returns:
            Dict: Complete response with answer, search results, and metadata
        """
        logger.info(f"Processing question: {query}")
        
        try:
            # Step 1: Search for relevant documents and chunks
            logger.info("Step 1: Searching for relevant documents...")
            search_results = await data_pull(
                query=query,
                use_reranker=use_reranker,
                subintent_top_k=subintent_top_k,
                document_top_k=document_top_k,
                chunk_top_k=chunk_top_k
            )
            
            if "error" in search_results:
                return {
                    "query": query,
                    "answer": f"Search error: {search_results['error']}",
                    "status": "error",
                    "search_results": search_results,
                    "chunks_used": [],
                    "context_length": 0
                }
            
            # Step 2: Extract and process chunks
            logger.info("Step 2: Extracting chunks from search results...")
            chunks = self.extract_chunks_text(search_results)
            
            if not chunks:
                return {
                    "query": query,
                    "answer": "Not enough information in the chunks to answer this question.",
                    "status": "no_chunks",
                    "search_results": search_results,
                    "chunks_used": [],
                    "context_length": 0
                }
            
            # Step 3: Create context from chunks
            logger.info("Step 3: Creating context from chunks...")
            context = self.create_context_from_chunks(chunks, max_context_chunks)
            
            if not context.strip():
                return {
                    "query": query,
                    "answer": "Not enough information in the chunks to answer this question.",
                    "status": "empty_context",
                    "search_results": search_results,
                    "chunks_used": chunks,
                    "context_length": 0
                }
            
            # Step 4: Generate answer using LLM
            logger.info("Step 4: Generating answer using LLM...")
            answer = await self.generate_answer(query, context)
            
            return {
                "query": query,
                "answer": answer,
                "status": "success",
                "search_results": search_results,
                "chunks_used": chunks[:max_context_chunks],
                "context_length": len(context),
                "total_chunks_found": len(chunks),
                "chunks_in_context": min(len(chunks), max_context_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "query": query,
                "answer": f"System error: {str(e)}",
                "status": "system_error",
                "search_results": {},
                "chunks_used": [],
                "context_length": 0
            }
    
    def display_qa_results(self, results: Dict[str, Any]):
        """
        Display the Q&A results in a formatted way.
        
        Args:
            results (Dict): Results from answer_question method
        """
        print(f"\n{'='*100}")
        print(f"DOCUMENT Q&A RESULTS")
        print(f"{'='*100}")
        
        print(f"\n🔍 QUERY: {results['query']}")
        print(f"📊 STATUS: {results['status'].upper()}")
        
        if results.get('chunks_used'):
            print(f"📝 CHUNKS USED: {results.get('chunks_in_context', 0)}/{results.get('total_chunks_found', 0)}")
            print(f"📏 CONTEXT LENGTH: {results.get('context_length', 0)} characters")
        
        print(f"\n{'='*50}")
        print(f"ANSWER:")
        print(f"{'='*50}")
        print(f"{results['answer']}")
        
        # Show chunk details if available
        chunks_used = results.get('chunks_used', [])
        if chunks_used:
            print(f"\n{'='*50}")
            print(f"CHUNKS USED IN CONTEXT:")
            print(f"{'='*50}")
            
            for chunk in chunks_used:
                print(f"\n📄 Chunk {chunk.get('rank', 'N/A')}:")
                print(f"   File: {chunk.get('file_id', 'Unknown')}")
                print(f"   Order: {chunk.get('chunk_order', 'N/A')}")
                print(f"   Score: {chunk.get('similarity_score', 0):.4f}")
                print(f"   Source: {chunk.get('source', 'unknown')}")
                if chunk.get('keywords'):
                    print(f"   Keywords: {chunk['keywords'][:3]}")
                print(f"   Text: {chunk.get('chunk_text', '')[:150]}...")


async def main():
    """
    Main function to demonstrate the Document Q&A System.
    """
    try:
        # Initialize the QA system
        qa_system = DocumentQASystem()
        
        # Test questions
        test_queries = [
            "What is the requirment for agronochain?",
        ]
        
        # Process each query
        for query in test_queries:
            print(f"\n{'='*120}")
            print(f"PROCESSING: {query}")
            print(f"{'='*120}")
            
            # Get answer using the QA system
            results = await qa_system.answer_question(
                query=query,
                use_reranker=True,
                max_context_chunks=8
            )
            
            # Display results
            qa_system.display_qa_results(results)
            
            # Add separator between queries
            print("\n" + "="*120)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")


async def ask_question(question: str, 
                      use_reranker: bool = True,
                      max_context_chunks: int = 10) -> Dict[str, Any]:
    """
    Convenient function to ask a single question and get an answer.
    
    Args:
        question (str): The question to ask
        use_reranker (bool): Whether to use reranker in search
        max_context_chunks (int): Maximum chunks to use for context
        
    Returns:
        Dict: Complete response with answer and metadata
    """
    qa_system = DocumentQASystem()
    return await qa_system.answer_question(
        query=question,
        use_reranker=use_reranker,
        max_context_chunks=max_context_chunks
    )


if __name__ == "__main__":
    # Example usage - you can modify this to ask specific questions
    asyncio.run(main())
    
    # Or ask a single question:
    # result = asyncio.run(ask_question("What is the Bill of Quantities and Prices for Furnishing 2018?"))
    # print(result['answer'])