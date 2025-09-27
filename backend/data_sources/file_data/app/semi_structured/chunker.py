import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid
from data_sources.file_data.app.semi_structured.utilites  import initialize_llm_gemini
# Initialize LLM (assuming this is your setup)
llm = initialize_llm_gemini()


def extract_markdown_from_file(file_path: str) -> str:
    """Extract markdown content from file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_file(file_path: str, chunk_size: int = 20000, chunk_overlap: int = 2000) -> List[str]:
    """
    Chunk a file into smaller parts using LangChain's RecursiveCharacterTextSplitter
    and log the number of chunks created to a text file.
    
    Args:
        file_path (str): Path to the file to be chunked.
        output_dir (str): Directory where chunk log file will be saved.
        chunk_size (int): Number of characters per chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.
    
    Returns:
        list: List of text chunks.
    """
    # Assume this function is already defined and working
    markdown_text = extract_markdown_from_file(file_path)
    
    # Split the text using LangChain's splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(markdown_text)
    output_dir = r"processing_results/"           # Directory to save chunks (optional)

    # Log chunk count to document_chunk_numbers.txt
    chunk_log_path = os.path.join(output_dir, "document_chunk_numbers.txt")
    with open(chunk_log_path, "a", encoding="utf-8") as f:
        f.write(f"{file_path} |Chunks: {len(chunks)}\n")

    
    return chunks

def clean_json_response(response_text: str) -> str:
    """
    Clean JSON response from LLM to handle common formatting issues.
    
    Args:
        response_text (str): Raw response from LLM
    
    Returns:
        str: Cleaned JSON string
    """
    # Remove markdown code blocks
    response_text = response_text.strip()
    if response_text.startswith('```json'):
        response_text = response_text[7:]
    if response_text.startswith('```'):
        response_text = response_text[3:]
    if response_text.endswith('```'):
        response_text = response_text[:-3]
    
    return response_text.strip()

def fix_json_string(json_str: str) -> str:
    """
    Advanced JSON string fixing with multiple strategies.
    
    Args:
        json_str (str): Potentially malformed JSON string
    
    Returns:
        str: Fixed JSON string
    """
    # Strategy 1: Fix common escape sequence issues
    def fix_escapes(s):
        # Fix unescaped quotes in strings
        s = re.sub(r'(?<!\\)"(?=(?:[^"\\]|\\.)*")', '\\"', s)
        # Fix unescaped backslashes
        s = re.sub(r'(?<!\\)\\(?!["\\\/bfnrtu])', '\\\\', s)
        # Fix newlines and tabs in strings
        s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
        return s
    
    # Strategy 2: Extract and fix the JSON structure
    def extract_json_structure(s):
        # Find the main JSON object
        brace_count = 0
        start_idx = s.find('{')
        if start_idx == -1:
            return s
        
        for i, char in enumerate(s[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return s[start_idx:i+1]
        return s
    
    # Strategy 3: Handle incomplete JSON
    def complete_json(s):
        if not s.strip().endswith('}'):
            # Count open braces vs close braces
            open_count = s.count('{')
            close_count = s.count('}')
            if open_count > close_count:
                s += '}' * (open_count - close_count)
        return s
    
    # Apply fixes in sequence
    json_str = extract_json_structure(json_str)
    json_str = fix_escapes(json_str)
    json_str = complete_json(json_str)
    
    return json_str

def process_chunk_with_llm(chunk: str, llm, max_retries: int = 3) -> Dict[str, Any]:
    """
    Process a single chunk with LLM and return structured data with robust error handling.
    
    Args:
        chunk (str): Text chunk to process
        llm: LLM instance
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        dict: Processed chunk metadata
    """
    prompt = get_processing_prompt()
    
    for attempt in range(max_retries):
        try:
            # Combine prompt with chunk
            full_prompt = f"{prompt}\n\nMarkdown content to process:\n{chunk}"
            
            response = llm.invoke(full_prompt)
            
            # Handle different response types
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Clean the response text
            response_text = clean_json_response(response_text)
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
                return result
            except json.JSONDecodeError as json_error:
                print(f"Attempt {attempt + 1}: JSON parsing error: {json_error}")
                print(f"Response text (first 500 chars): {response_text[:500]}...")
                
                # Try to fix the JSON
                fixed_json = fix_json_string(response_text)
                
                try:
                    result = json.loads(fixed_json)
                    print(f"Successfully fixed JSON on attempt {attempt + 1}")
                    return result
                except json.JSONDecodeError:
                    print(f"Failed to fix JSON on attempt {attempt + 1}")
                    
                    # If this is not the last attempt, regenerate response
                    if attempt < max_retries - 1:
                        print(f"Regenerating LLM response (attempt {attempt + 2}/{max_retries})")
                        # Add specific instruction to fix JSON issues
                        enhanced_prompt = f"""{prompt}

CRITICAL: The previous response had JSON formatting issues. Please ensure:
1. All strings are properly escaped (use \\" for quotes within strings)
2. No unescaped newlines or tabs in string values
3. Complete JSON structure with proper closing braces
4. Return ONLY valid JSON with no additional text

Markdown content to process:
{chunk}"""
                        full_prompt = enhanced_prompt
                        continue
                    else:
                        print("Max retries reached, returning empty result")
                        return create_empty_result()
                        
        except Exception as e:
            print(f"Attempt {attempt + 1}: Error processing chunk: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                continue
            else:
                print("Max retries reached, returning empty result")
                return create_empty_result()
    
    # Should not reach here, but just in case
    return create_empty_result()

def get_processing_prompt() -> str:
    """Return the processing prompt for LLM."""
    return """
Markdown Table Processing Instructions (JSON Output)
You are a specialized assistant for processing markdown files containing tables. When given a markdown file, perform the following tasks systematically and return results in JSON format:

IMPORTANT: Return ONLY valid JSON. Do not include any explanation, markdown formatting, or code blocks. Ensure all strings are properly escaped with double quotes and backslashes.

Processing Tasks
1. **Table Identification**
   * Scan the markdown content and identify all tables
   * If multiple tables exist, process each one separately

2. **Extract Last 2-3 Lines for Search Chunking**
   * Extract the last 2-3 data rows from each table
   * Return the rows exactly as they appear in the markdown table
   * Do not reformat or restructure the data
   * Preserve original markdown table formatting

3. **Extract Meaningful Keywords**
   * Identify and extract all meaningful keywords including:
      * Person names
      * Organization names
      * Location names
      * Product names
      * Technical terms
      * Important identifiers (IDs, codes, etc.)
   * Return as an array of keywords

4. **Date Range Extraction**
   * Scan all table data for dates in any format
   * Identify the earliest and latest dates
   * Convert to ISO format (YYYY-MM-DD)
   * Return start and end dates
   * If no dates found, return null for both

5. **Generate Table Description**
   * Create a comprehensive description that includes:
      * Purpose/context of the table
      * Key data points and patterns
      * Notable trends or insights
      * Data types and ranges
      * Any missing or incomplete data
   * Write in 2-3 paragraphs, focusing on meaningful insights
   * Ensure all text is properly escaped for JSON (use \\" for quotes, \\n for newlines)

6. **Extract Column Names**
   * List all column headers exactly as they appear 
   * Note any merged headers or sub-headers
   * Return as an array

7. **Generate Title**
   * Create a concise, descriptive title for the content
   * Should summarize the main topic or purpose

Must follow the JSON output format exactly as specified below. If anything is missing, return an empty JSON object with the required structure.

JSON Output Format (return exactly this structure):
{
  "title": "string",
  "table_analysis": {
    "tables": [
      {
        "last_lines_for_chunking": [
          "string (raw markdown table row)"
        ],
        "keywords": ["string"],
        "date_range": {
          "start_date": null,
          "end_date": null
        },
        "table_description": "string (comprehensive description with proper escaping)",
        "column_names": ["string"]
      }
    ]
  }
}

CRITICAL: 
- Return ONLY the JSON object
- No additional text, explanations, or markdown formatting
- Properly escape all strings (use \\" for quotes, \\n for newlines, \\\\ for backslashes)
- Ensure complete JSON structure with all required fields
"""

def create_empty_result() -> Dict[str, Any]:
    """Create empty result structure when processing fails."""
    return {
        "title": "",
        "table_analysis": {
            "tables": []
        }
    }

def check_missing_column_names(current_result: Dict[str, Any]) -> List[int]:
    """
    Check which tables are missing column names.
    
    Args:
        current_result (dict): Current chunk processing result
    
    Returns:
        list: List of table indices that are missing column names
    """
    missing_tables = []
    
    if "table_analysis" in current_result:
        tables = current_result["table_analysis"].get("tables", [])
        for idx, table in enumerate(tables):
            column_names = table.get("column_names", [])
            if not column_names or all(not name.strip() for name in column_names):
                missing_tables.append(idx)
    
    return missing_tables

def extract_column_names_from_previous_chunks(previous_chunks: List[Dict[str, Any]], 
                                            table_idx: int) -> List[str]:
    """
    Extract column names from previous chunks for a specific table.
    
    Args:
        previous_chunks (list): List of previous chunk results
        table_idx (int): Index of the table to find column names for
    
    Returns:
        list: Column names found in previous chunks
    """
    for chunk_result in reversed(previous_chunks):  # Search from most recent
        if "table_analysis" in chunk_result:
            tables = chunk_result["table_analysis"].get("tables", [])
            if table_idx < len(tables):
                column_names = tables[table_idx].get("column_names", [])
                if column_names and any(name.strip() for name in column_names):
                    return column_names
    
    return []

def fix_missing_column_names(current_result: Dict[str, Any], 
                           previous_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fix missing column names by finding them in previous chunks.
    
    Args:
        current_result (dict): Current chunk processing result
        previous_chunks (list): List of previous chunk results
    
    Returns:
        dict: Updated result with fixed column names
    """
    missing_table_indices = check_missing_column_names(current_result)
    
    if not missing_table_indices:
        return current_result
    
    # Create a copy to avoid modifying the original
    updated_result = json.loads(json.dumps(current_result))
    
    for table_idx in missing_table_indices:
        column_names = extract_column_names_from_previous_chunks(previous_chunks, table_idx)
        
        if column_names:
            # Find and update the table
            tables = updated_result["table_analysis"]["tables"]
            if table_idx < len(tables):
                tables[table_idx]["column_names"] = column_names
                print(f"Fixed missing column names for table {table_idx}")
    
    return updated_result

def generate_combined_title_with_llm(processed_chunks: List[Dict[str, Any]], llm) -> str:
    """
    Generate a combined title using LLM from all chunk titles.
    
    Args:
        processed_chunks (list): List of processed chunks
        llm: LLM instance
    
    Returns:
        str: LLM-generated combined title
    """
    titles = []
    for chunk in processed_chunks:
        title = chunk.get("title", "").strip()
        if title and title not in titles:
            titles.append(title)
    
    if not titles:
        return "Processed Markdown Document"
    
    # If there's only one unique title, use it
    if len(titles) == 1:
        return titles[0]
    
    # Use LLM to generate combined title
    title_prompt = f"""
Create a comprehensive, concise title that captures the essence of this document based on these individual chunk titles:

Chunk Titles:
{chr(10).join(f"- {title}" for title in titles)}

Requirements:
- Create ONE unified title that represents the overall document content
- Keep it under 100 characters
- Make it descriptive and professional
- Focus on the main theme or subject matter
- Return ONLY the title, no additional text or explanation

Title:"""

    try:
        response = llm.invoke(title_prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            combined_title = response.content.strip()
        elif hasattr(response, 'text'):
            combined_title = response.text.strip()
        else:
            combined_title = str(response).strip()
        
        # Clean up the response
        combined_title = combined_title.replace('"', '').replace("'", "")
        
        return combined_title if combined_title else " | ".join(titles[:3])
        
    except Exception as e:
        print(f"Error generating combined title with LLM: {e}")
        return " | ".join(titles[:3])  # Fallback to simple joining

def generate_combined_description_with_llm(processed_chunks: List[Dict[str, Any]], llm) -> str:
    """
    Generate a combined description using LLM from all chunk descriptions.
    
    Args:
        processed_chunks (list): List of processed chunks
        llm: LLM instance
    
    Returns:
        str: LLM-generated combined description
    """
    descriptions = []
    titles = []
    
    for chunk in processed_chunks:
        title = chunk.get("title", "").strip()
        if title:
            titles.append(title)
            
        tables = chunk.get("table_analysis", {}).get("tables", [])
        for table in tables:
            desc = table.get("table_description", "").strip()
            if desc:
                descriptions.append(desc)
    
    if not descriptions:
        return "This document contains processed markdown tables with extracted metadata."
    
    # Use LLM to generate combined description
    description_prompt = f"""
Create a comprehensive description that synthesizes and summarizes the content of this document based on the individual chunk descriptions below.

Individual Chunk Titles:
{chr(10).join(f"- {title}" for title in titles if title)}

Individual Table Descriptions:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Requirements:
- Create ONE cohesive description that captures the overall document content
- Synthesize information from all descriptions rather than just concatenating them
- Focus on main themes, patterns, and insights across the entire document
- Keep it between 200-400 words
- Write in paragraph form (2-3 paragraphs)
- Be professional and informative
- Highlight key data types, time periods, and subject matter
- Return ONLY the description, no additional text or explanation

Description:"""

    try:
        response = llm.invoke(description_prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            combined_description = response.content.strip()
        elif hasattr(response, 'text'):
            combined_description = response.text.strip()
        else:
            combined_description = str(response).strip()
        
        return combined_description if combined_description else "\n\n".join(descriptions)
        
    except Exception as e:
        print(f"Error generating combined description with LLM: {e}")
        return "\n\n".join(descriptions)  # Fallback to simple joining

def process_file_with_chunking(file_path: str, chunk_size: int = 20000, chunk_overlap: int = 1000, llm=None) -> Dict[str, Any]:
    """
    Main function to process file with chunking and metadata extraction.
    
    Args:
        file_path (str): Path to the markdown file
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Number of characters to overlap between chunks
        llm: LLM instance for processing
    
    Returns:
        dict: Complete processing result with file metadata
    """
    if not llm:
        raise ValueError("LLM instance is required")
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Step 1: Create chunks using LangChain
    chunks = chunk_file(file_path, chunk_size, chunk_overlap)
    processed_chunks = []
    
    print(f"Processing {len(chunks)} chunks...")
    
    # Step 2: Process each chunk
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Generate unique chunk ID
        chunk_id = str(uuid.uuid4())
        
        # Process with LLM (now with retry logic)
        chunk_result = process_chunk_with_llm(chunk, llm, max_retries=3)
        
        # Step 3: Fix missing column names using previous chunks
        if processed_chunks:  # Only if we have previous chunks
            chunk_result = fix_missing_column_names(chunk_result, processed_chunks)
        
        # Step 4: Add chunk metadata
        chunk_result["chunk_id"] = chunk_id
        chunk_result["chunk_text"] = chunk
        chunk_result["chunk_size"] = len(chunk)
        
        processed_chunks.append(chunk_result)
    
    # Step 5: Generate combined metadata using LLM
    print("Generating combined title and description with LLM...")
    combined_title = generate_combined_title_with_llm(processed_chunks, llm)
    combined_description = generate_combined_description_with_llm(processed_chunks, llm)
    
    # Step 6: Create final result structure (without document_summary)
    result = {
        "file_id": file_id,
        "file_path": file_path,
        "combined_title": combined_title,
        "combined_description": combined_description,
        "total_chunks": len(processed_chunks),
        "processed_at": datetime.now().isoformat(),
        "chunks": processed_chunks
    }
    
    return result

def save_results_to_file(processing_result: Dict[str, Any], output_file: str):
    """
    Save processing result to a JSON file.
    
    Args:
        processing_result (dict): Complete processing result
        output_file (str): Path to output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processing_result, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def get_processing_summary(processing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of the processing result.
    
    Args:
        processing_result (dict): Complete processing result
    
    Returns:
        dict: Summary statistics
    """
    chunks = processing_result.get("chunks", [])
    total_chunks = len(chunks)
    total_tables = sum(len(chunk.get("table_analysis", {}).get("tables", [])) for chunk in chunks)
    
    all_keywords = []
    all_dates = []
    
    for chunk in chunks:
        tables = chunk.get("table_analysis", {}).get("tables", [])
        for table in tables:
            all_keywords.extend(table.get("keywords", []))
            date_range = table.get("date_range", {})
            if date_range.get("start_date"):
                all_dates.append(date_range["start_date"])
            if date_range.get("end_date"):
                all_dates.append(date_range["end_date"])
    
    unique_keywords = list(set(all_keywords))
    
    return {
        "file_id": processing_result.get("file_id"),
        "combined_title": processing_result.get("combined_title"),
        "total_chunks": total_chunks,
        "total_tables": total_tables,
        "unique_keywords_count": len(unique_keywords),
        "unique_keywords": unique_keywords[:20],  # Top 20 keywords
        "date_range": {
            "earliest": min(all_dates) if all_dates else None,
            "latest": max(all_dates) if all_dates else None
        },
        "processed_at": processing_result.get("processed_at")
    }

# # Example usage
# def main():
#     """Example usage of the markdown processing system."""
    
#     # Initialize your LLM here
#     # llm = initialize_llm_gemini()
    
#     file_path = "/Users/nilab/Desktop/projects/Knowladge-Base/output.md"
#     output_file = "processed_file.json"
    
#     try:
#         # Process the file
#         processing_result = process_file_with_chunking(file_path, chunk_size=20000, chunk_overlap=1000, llm=llm)
        
#         # Save results
#         save_results_to_file(processing_result, output_file)
        
#         # Generate summary
#         summary = get_processing_summary(processing_result)
#         print("Processing Summary:")
#         print(json.dumps(summary, indent=2))
        
#     except Exception as e:
#         print(f"Error processing file: {e}")

# if __name__ == "__main__":
#     main()