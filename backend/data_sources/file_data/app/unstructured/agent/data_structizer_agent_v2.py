import os
import json
from time import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from datetime import datetime
import re
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from enum import Enum

# Import the new config loader
from data_sources.file_data.app.unstructured.agent.config_loader import get_database_config, DatabaseConfig

# Load environment variables
load_dotenv(override=True)

# Retrieve API key and model name from environment variables
gemini_apikey = os.getenv("google_api_key")
if gemini_apikey is None:
    print("Warning: 'google_api_key' not found in environment variables.")

gemini_model_name = os.getenv("google_gemini_name", "gemini-1.5-pro")
if gemini_model_name is None:
    print("Warning: 'google_gemini_name' not found in environment variables, using default 'gemini-1.5-pro'.")

def initialize_llm_gemini(api_key: str = gemini_apikey, temperature: int = 0, model: str = gemini_model_name) -> ChatGoogleGenerativeAI:
    """
    Initialize and return the ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)  

import sys
from pathlib import Path

# Add /Users/nilab/Desktop/projects/Knowladge-Base/unstructured/ to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from parser.custom_perser import SmartDocumentProcessor
from chunker.chunker_V1 import SmartTextChunker

llm = initialize_llm_gemini()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects."""
    
    def default(self, obj):
        # Handle Enum objects
        if isinstance(obj, Enum):
            return obj.value
        
        # Handle any object with a __dict__ attribute
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        
        # Handle any object that has a string representation
        if hasattr(obj, '__str__'):
            return str(obj)
        
        # For any other non-serializable object, convert to string
        try:
            return super().default(obj)
        except TypeError:
            return f"<Non-serializable {type(obj).__name__} object>"

def safe_serialize_data(data: Any) -> Any:
    """
    Recursively clean data to make it JSON serializable.
    """
    if isinstance(data, dict):
        return {key: safe_serialize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [safe_serialize_data(item) for item in data]
    elif isinstance(data, tuple):
        return [safe_serialize_data(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    elif hasattr(data, '__dict__'):
        # Convert objects with __dict__ to dictionary
        return safe_serialize_data(data.__dict__)
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        # Convert anything else to string
        return str(data)

@dataclass
class FileMetadata:
    """Metadata for the processed file."""
    file_id: str
    file_name: str
    file_type: str
    file_path: str
    extracted_text: str
    full_summary: str
    title: str
    keywords: List[str]
    date_range: Optional[Dict[str, str]]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    processing_timestamp: str
    file_description: Optional[str] = None  # New field for file description
    table_name: Optional[str] = None  # New field for table name
    user_id: Optional[str] = None  # New field for user id

@dataclass
class ChunkMetadata:
    """Metadata for each chunk."""
    file_id: str
    chunk_id: str
    chunk_text: str
    summary: str
    title: str
    keywords: List[str]
    date_range: Optional[Dict[str, str]]  # {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    # Add page information fields
    page_numbers: Optional[List[int]] = None  # List of page numbers in this chunk
    page_range: Optional[Dict[str, int]] = None  # {"start": 1, "end": 3}
    word_count: Optional[int] = None  # Number of words in chunk
    chunk_boundaries: Optional[Dict[str, int]] = None  # {"start": 1, "end": 4}

class LLMProcessor:
    """Handles all LLM interactions for generating summaries, titles, and keywords."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def generate_full_document_metadata(self, text: str, file_name: str) -> Dict[str, Any]:
        """Generate summary, title, keywords, and date range for the full document."""
        prompt = f"""
        Analyze the following document and provide:
        1. A comprehensive summary (3-5 sentences)
        2. A descriptive title
        3. Keywords including: names, places, organizations, key concepts, entities
        4. Date range if any dates are mentioned in the document (return min and max dates in YYYY-MM-DD format)
        
        Document filename: {file_name}
        Document content: {text[:99000]}...  # First 99000 chars to avoid token limits
        
        Return the response in JSON format:
        {{
            "summary": "...",
            "title": "...",
            "keywords": ["keyword1", "keyword2", ...],
            "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null if no dates found
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            # Handle both string and object responses
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Clean the response text to extract JSON
            response_text = response_text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            try:
                parsed_response = json.loads(response_text)
                # Ensure all required fields are present
                required_fields = ["summary", "title", "keywords", "date_range"]
                for field in required_fields:
                    if field not in parsed_response:
                        raise ValueError(f"Missing required field: {field}")
                
                # Validate and clean data types
                if not isinstance(parsed_response["summary"], str):
                    parsed_response["summary"] = str(parsed_response["summary"])
                if not isinstance(parsed_response["title"], str):
                    parsed_response["title"] = str(parsed_response["title"])
                if not isinstance(parsed_response["keywords"], list):
                    parsed_response["keywords"] = []
                if parsed_response["date_range"] is not None and not isinstance(parsed_response["date_range"], dict):
                    parsed_response["date_range"] = None
                
                return parsed_response
                
            except json.JSONDecodeError as je:
                print(f"Failed to parse JSON for document {file_name}: {je}")
                print(f"Raw response: {response_text}")
                raise ValueError(f"Invalid JSON response from LLM")
                
        except Exception as e:
            print(f"Error generating full document metadata: {e}")
            return {
                "summary": f"Document summary unavailable due to processing error: {str(e)}",
                "title": f"Document: {file_name}",
                "keywords": [],
                "date_range": None
            }
    
    def generate_chunk_metadata(self, chunk_text: str, chunk_id: str) -> Dict[str, Any]:
        """Generate summary, title, keywords, and date range for a text chunk."""
        prompt = f"""
        Analyze the following text chunk and provide:
        1. A concise summary (2-3 sentences)
        2. A descriptive title for this chunk
        3. Keywords including: names, places, organizations, key concepts, entities
        4. Date range if any dates are mentioned (return min and max dates in YYYY-MM-DD format)
        
        Chunk ID: {chunk_id}
        Chunk content: {chunk_text}
        
        Return the response in JSON format:
        {{
            "summary": "...",
            "title": "...",
            "keywords": ["keyword1", "keyword2", ...],
            "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null if no dates found
        }}
        """
        
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                response = self.llm.invoke(prompt)
                # Handle both string and object responses
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                # Clean the response text to extract JSON
                response_text = response_text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                
                # Parse and validate the response
                try:
                    parsed_response = json.loads(response_text)
                    # Ensure all required fields are present
                    required_fields = ["summary", "title", "keywords", "date_range"]
                    for field in required_fields:
                        if field not in parsed_response:
                            raise ValueError(f"Missing required field: {field}")
                    
                    # Validate data types
                    if not isinstance(parsed_response["summary"], str):
                        parsed_response["summary"] = str(parsed_response["summary"])
                    if not isinstance(parsed_response["title"], str):
                        parsed_response["title"] = str(parsed_response["title"])
                    if not isinstance(parsed_response["keywords"], list):
                        parsed_response["keywords"] = []
                    if parsed_response["date_range"] is not None and not isinstance(parsed_response["date_range"], dict):
                        parsed_response["date_range"] = None
                    
                    return parsed_response
                    
                except json.JSONDecodeError as je:
                    print(f"Failed to parse JSON for chunk {chunk_id}: {je}")
                    raise ValueError(f"Invalid JSON response: {response_text}")
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a rate limit error (429)
                if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                    retry_count += 1
                    if retry_count < max_retries:
                        retry_delay = 60  # Default delay
                        if "retry_delay" in error_str and "seconds:" in error_str:
                            try:
                                delay_part = error_str.split("seconds:")[1].split("}")[0].strip()
                                retry_delay = int(delay_part) + 5  # Add 5 seconds buffer
                            except:
                                retry_delay = 60  # Fallback to 60 seconds
                        
                        print(f"Rate limit hit for chunk {chunk_id}. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"Max retries reached for chunk {chunk_id}. Rate limit error: {last_error}")
                        break
                else:
                    # Non-rate-limit error, don't retry
                    print(f"Error generating chunk metadata for {chunk_id}: {last_error}")
                    break
        
        # Return fallback response if all retries failed
        return {
            "summary": f"Chunk summary unavailable due to error: {str(last_error)}",
            "title": f"Chunk {chunk_id}",
            "keywords": [],
            "date_range": None
        }

class DocumentProcessingPipeline:
    """Main pipeline for document processing."""
    
    def __init__(self, document_processor, text_chunker, llm_client, max_workers: int = 4):
        self.document_processor = document_processor
        self.text_chunker = text_chunker
        self.llm_processor = LLMProcessor(llm_client)
        self.max_workers = max_workers
    
    def generate_file_id(self) -> str:
        """Generate unique file ID."""
        return str(uuid.uuid4())
    
    def extract_file_info(self, file_path: str) -> Tuple[str, str]:
        """Extract file name and type from path."""
        path = Path(file_path)
        file_name = path.name
        file_type = path.suffix.lower().lstrip('.')
        return file_name, file_type
    
    def process_document(self, file_path: str, output_dir: Optional[str] = None, 
                        file_description: Optional[str] = None, 
                        table_name: Optional[str] = None, 
                        user_id: Optional[str] = None) -> Tuple[FileMetadata, List[ChunkMetadata]]:
        """
        Main processing pipeline.
        
        Args:
            file_path: Path to the document to process
            output_dir: Optional directory to save chunks
            file_description: Optional description for the file
            table_name: Optional table name for the file
            user_id: Optional user id for the file
            
        Returns:
            Tuple of (FileMetadata, List[ChunkMetadata])
        """
        
        # Step 1: Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Step 2: Generate file metadata
        file_id = self.generate_file_id()
        file_name, file_type = self.extract_file_info(file_path)
        
        print(f"Processing file: {file_name} (ID: {file_id})")
        
        # Step 3: Extract text from document
        print("Extracting text from document...")
        try:
            extracted_text = self.document_processor.process_document(file_path, output_dir)
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")
        
        print(f"Extracted {len(extracted_text):,} characters")
        
        # Step 4: Create threading for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit full document analysis
            full_doc_future = executor.submit(
                self.llm_processor.generate_full_document_metadata,
                extracted_text,
                file_name
            )
            
            # Submit chunking task
            chunking_future = executor.submit(
                self.text_chunker.chunk_text,
                extracted_text
            )
            
            # Wait for chunking to complete
            print("Creating text chunks...")
            chunks = chunking_future.result()
            print(f"Created {len(chunks)} chunks")
            
            # Submit chunk analysis tasks
            chunk_futures = {}
            for i, chunk in enumerate(chunks):
                # Handle both string chunks and chunk objects
                if isinstance(chunk, str):
                    chunk_text = chunk
                    page_numbers = None
                    page_range = None
                    word_count = len(chunk.split())
                    chunk_boundaries = None
                else:
                    # Extract information from chunk object
                    chunk_text = getattr(chunk, 'content', str(chunk))
                    page_numbers = getattr(chunk, 'page_numbers', None)
                    word_count = getattr(chunk, 'word_count', len(chunk_text.split()))
                    
                    # Create page_range from page_numbers if available
                    if page_numbers and len(page_numbers) > 0:
                        page_range = {
                            "start": min(page_numbers),
                            "end": max(page_numbers)
                        }
                    else:
                        page_range = None
                        
                    # Extract boundary info if available
                    boundary_info = getattr(chunk, 'boundary_info', None)
                    if boundary_info:
                        chunk_boundaries = {
                            "starts_with_page": boundary_info.get('starts_with_page'),
                            "ends_with_page": boundary_info.get('ends_with_page'),
                            "contains_tables": boundary_info.get('contains_tables', False),
                            "contains_figures": boundary_info.get('contains_figures', False),
                            "contains_chapters": boundary_info.get('contains_chapters', False),
                            "segment_titles": boundary_info.get('segment_titles', []),
                            "overlap_info": boundary_info.get('overlap_info', '')
                        }
                    else:
                        chunk_boundaries = None
                
                chunk_id = f"{file_id}_chunk_{i+1:03d}"
                future = executor.submit(
                    self.llm_processor.generate_chunk_metadata,
                    chunk_text,
                    chunk_id
                )
                chunk_futures[chunk_id] = (future, chunk_text, page_numbers, page_range, word_count, chunk_boundaries)

                        
            # Wait for full document analysis
            print("Analyzing full document...")
            full_doc_metadata = full_doc_future.result()
            
            # Create FileMetadata object with date range
            file_metadata = FileMetadata(
                file_id=file_id,
                file_name=file_name,
                file_type=file_type,
                file_path=file_path,
                extracted_text=extracted_text,
                full_summary=full_doc_metadata.get("summary", ""),
                title=full_doc_metadata.get("title", file_name),
                keywords=full_doc_metadata.get("keywords", []),
                date_range=full_doc_metadata.get("date_range"),
                processing_timestamp=datetime.now().isoformat(),
                file_description=file_description,
                table_name=table_name,
                user_id=user_id
            )
            
            print(f"🔍 DEBUG: Created FileMetadata with file_description='{file_description}', table_name='{table_name}', user_id='{user_id}'")
            
            # Process chunk results
            print("Analyzing chunks...")
            chunk_metadata_list = []

            for chunk_id, (future, chunk_text, page_numbers, page_range, word_count, chunk_boundaries) in chunk_futures.items():
                try:
                    chunk_metadata_dict = future.result()
                    
                    chunk_metadata = ChunkMetadata(
                        file_id=file_id,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                        summary=chunk_metadata_dict.get("summary", ""),
                        title=chunk_metadata_dict.get("title", f"Chunk {chunk_id}"),
                        keywords=chunk_metadata_dict.get("keywords", []),
                        date_range=chunk_metadata_dict.get("date_range"),
                        page_numbers=page_numbers,
                        page_range=page_range,
                        word_count=word_count,
                        chunk_boundaries=chunk_boundaries
                    )
                    
                    chunk_metadata_list.append(chunk_metadata)
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_id}: {e}")
                    # Create fallback chunk metadata with page info if available
                    chunk_metadata = ChunkMetadata(
                        file_id=file_id,
                        chunk_id=chunk_id,
                        chunk_text=chunk_text,
                        summary=f"Error processing chunk: {str(e)}",
                        title=f"Chunk {chunk_id}",
                        keywords=[],
                        date_range=None,
                        page_numbers=page_numbers,
                        page_range=page_range,
                        word_count=word_count,
                        chunk_boundaries=chunk_boundaries
                    )
                    chunk_metadata_list.append(chunk_metadata)
        
        print(f"Processing complete! Generated metadata for {len(chunk_metadata_list)} chunks")
        return file_metadata, chunk_metadata_list
    
    def save_results(self, file_metadata: FileMetadata, chunk_metadata_list: List[ChunkMetadata], 
                    output_dir: str = "processing_results") -> Dict[str, str]:
        """Save processing results to JSON files."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save file metadata
        file_metadata_path = os.path.join(output_dir, f"{file_metadata.file_id}_file_metadata.json")
        with open(file_metadata_path, 'w', encoding='utf-8') as f:
            # Use safe serialization
            file_data = safe_serialize_data(asdict(file_metadata))
            json.dump(file_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        # Save chunk metadata
        chunks_metadata_path = os.path.join(output_dir, f"{file_metadata.file_id}_chunks_metadata.json")
        chunks_data = [safe_serialize_data(asdict(chunk)) for chunk in chunk_metadata_list]
        with open(chunks_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
        
        # Save summary report
        summary_path = os.path.join(output_dir, f"{file_metadata.file_id}_summary_report.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"DOCUMENT PROCESSING SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"File: {file_metadata.file_name}\n")
            f.write(f"Type: {file_metadata.file_type}\n")
            f.write(f"File ID: {file_metadata.file_id}\n")
            f.write(f"Title: {file_metadata.title}\n")
            f.write(f"Processing Time: {file_metadata.processing_timestamp}\n")
            f.write(f"Text Length: {len(file_metadata.extracted_text):,} characters\n")
            f.write(f"Number of Chunks: {len(chunk_metadata_list)}\n")
            
            # Add date range information for the full document
            if file_metadata.date_range:
                f.write(f"Document Date Range: {file_metadata.date_range['start']} to {file_metadata.date_range['end']}\n")
            else:
                f.write(f"Document Date Range: No dates found\n")
            f.write(f"\n")
            
            f.write(f"FULL DOCUMENT SUMMARY:\n")
            f.write(f"{file_metadata.full_summary}\n\n")
            
            f.write(f"KEYWORDS: {', '.join(file_metadata.keywords)}\n\n")
            
            f.write(f"CHUNK SUMMARIES:\n")
            f.write(f"{'-'*30}\n")
            for i, chunk in enumerate(chunk_metadata_list, 1):
                f.write(f"\nChunk {i}: {chunk.title}\n")
                f.write(f"Summary: {chunk.summary}\n")
                f.write(f"Keywords: {', '.join(chunk.keywords)}\n")
                if chunk.page_numbers:
                    f.write(f"Pages: {chunk.page_numbers}\n")
                if chunk.page_range:
                    f.write(f"Page Range: {chunk.page_range['start']} to {chunk.page_range['end']}\n")
                if chunk.word_count:
                    f.write(f"Word Count: {chunk.word_count}\n")
                if chunk.date_range:
                    f.write(f"Date Range: {chunk.date_range['start']} to {chunk.date_range['end']}\n")
                else:
                    f.write(f"Date Range: No dates found\n")
        
        return {
            "file_metadata": file_metadata_path,
            "chunks_metadata": chunks_metadata_path,
            "summary_report": summary_path
        }

def create_pipeline(document_processor, text_chunker, llm_client, max_workers: int = 4) -> DocumentProcessingPipeline:
    """Factory function to create a document processing pipeline."""
    return DocumentProcessingPipeline(
        document_processor=document_processor,
        text_chunker=text_chunker,
        llm_client=llm_client,
        max_workers=max_workers
    )

def main_pipeline_example():
    """Example usage of the document processing pipeline."""
    
    # Initialize your components (replace with your actual implementations)
    try:
        # Initialize document processor
        document_processor = SmartDocumentProcessor(
            max_pages_per_chunk=5,
            boundary_sentences=3,
            boundary_table_rows=3
        )
        
        # Initialize text chunker
        text_chunker = SmartTextChunker(
            target_pages_per_chunk=3,
            overlap_pages=1,
            max_pages_per_chunk=3,
            min_pages_per_chunk=1,
            respect_boundaries=True
        )
        
        # Initialize LLM client
        llm_client = llm  # Replace with your actual LLM client
        
        # Create pipeline
        pipeline = create_pipeline(document_processor, text_chunker, llm_client)
        
        # Process document
        file_path = "/Users/nilab/Desktop/projects/Knowladge-Base/Md__Nabil_Rahman_Khan_CV.PDF"  # Update this path
        
        print("Starting document processing pipeline...")
        file_metadata, chunk_metadata_list = pipeline.process_document(file_path)
        
        # Save results
        saved_files = pipeline.save_results(file_metadata, chunk_metadata_list)
        
        print("\nProcessing completed successfully!")
        print("Saved files:")
        for file_type, path in saved_files.items():
            print(f"  {file_type}: {path}")
        
        # Display summary
        print(f"\nRESULTS SUMMARY:")
        print(f"File ID: {file_metadata.file_id}")
        print(f"Title: {file_metadata.title}")
        print(f"Chunks created: {len(chunk_metadata_list)}")
        print(f"Total keywords: {len(set(file_metadata.keywords + [kw for chunk in chunk_metadata_list for kw in chunk.keywords]))}")
        
        # Display date range information
        if file_metadata.date_range:
            print(f"Document Date Range: {file_metadata.date_range['start']} to {file_metadata.date_range['end']}")
        else:
            print(f"Document Date Range: No dates found")
        
        return file_metadata, chunk_metadata_list
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_pipeline_example()