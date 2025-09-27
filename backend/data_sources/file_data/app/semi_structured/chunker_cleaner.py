#chunker_cleaner.py
import json
import uuid
from typing import List, Dict, Any
from data_sources.file_data.app.semi_structured.utilites  import initialize_llm_gemini

# Initialize LLM
llm = initialize_llm_gemini()

def load_json_data(file_path: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load JSON data from file and extract chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle the nested structure from the main processing script
    if isinstance(data, dict) and "chunks" in data:
        # Extract chunks from the nested structure
        chunks = data["chunks"]
        metadata = {k: v for k, v in data.items() if k != "chunks"}
        return metadata, chunks
    elif isinstance(data, list):
        # Handle direct list of chunks (backward compatibility)
        return {}, data
    else:
        raise ValueError("Invalid JSON structure. Expected either a dict with 'chunks' key or a list of chunks.")

def save_json_data(metadata: Dict[str, Any], chunks: List[Dict[str, Any]], file_path: str):
    """Save JSON data to file with original structure."""
    if metadata:
        # Reconstruct the original nested structure
        output_data = metadata.copy()
        output_data["chunks"] = chunks
        output_data["total_chunks"] = len(chunks)  # Update chunk count
    else:
        # Save as list if no metadata
        output_data = chunks
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def get_column_comparison_prompt(previous_columns: List[str], previous_chunk_text: str, current_chunk_text: str) -> str:
    """Generate improved prompt for LLM to compare column names and table continuity."""
    return f"""
You are a table analysis expert. I need you to carefully compare two document chunks and determine whether they belong to the same table.

Follow these steps:
1. Look at the **column names from the previous chunk**: {previous_columns}
2. Read the **text of the previous chunk**:
\"\"\"
{previous_chunk_text}
\"\"\"
3. Read the **text of the current chunk**:
\"\"\"
{current_chunk_text}
\"\"\"
4. Decide whether the current chunk continues the same table or starts a new one.
5. If it continues, reuse the previous column names.
6. If it starts a new table, analyze and extract new column names based on patterns in the current chunk text (like table headers, data formats, etc.).

Respond with ONLY a JSON object in this exact format:
{{
  "is_same_table": true or false,
  "suggested_column_names": ["column1", "column2", "..."],
  "confidence": "high", "medium" or "low"
}}

**Rules:**
- Do NOT include any explanation or additional text — only the JSON object.
- Assume that chunks are sequential. Tables might span multiple chunks if data continues.
- If uncertain, prefer setting "is_same_table" to false and extract new columns.
- Confidence should reflect how certain you are about your decision based on column alignment and data patterns.

Important: your response must be valid JSON parsable by a JSON parser.
"""

def check_column_names_with_llm(previous_columns: List[str], previous_chunk_text: str, current_chunk_text: str) -> Dict[str, Any]:
    """Use LLM to check if current chunk has same table as previous chunk."""
    prompt = get_column_comparison_prompt(previous_columns, previous_chunk_text, current_chunk_text)
    
    try:
        response = llm.invoke(prompt)
        
        # Handle different response types
        if hasattr(response, 'content'):
            response_text = response.content
        elif hasattr(response, 'text'):
            response_text = response.text
        else:
            response_text = str(response)
        
        # Clean response
        response_text = response_text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse JSON
        result = json.loads(response_text.strip())
        return result
        
    except Exception as e:
        print(f"Error in LLM analysis: {e}")
        return {
            "is_same_table": False,
            "suggested_column_names": [],
            "confidence": "low"
        }

def extract_column_names_from_tables(chunk_data: Dict[str, Any]) -> List[str]:
    """Extract column names from chunk data."""
    column_names = []
    
    if "table_analysis" in chunk_data:
        tables = chunk_data["table_analysis"].get("tables", [])
        for table in tables:
            names = table.get("column_names", [])
            if names:
                column_names.extend(names)
    
    return list(set(column_names))  # Remove duplicates

def update_chunk_with_same_table_info(chunk_data: Dict[str, Any], 
                                    previous_chunk: Dict[str, Any], 
                                    table_group_id: str) -> Dict[str, Any]:
    """Update chunk data when it's from the same table as previous chunk."""
    
    # Add table group ID
    chunk_data["table_group_id"] = table_group_id
    
    # Update column names, title, and description for each table
    if "table_analysis" in chunk_data and "table_analysis" in previous_chunk:
        current_tables = chunk_data["table_analysis"].get("tables", [])
        previous_tables = previous_chunk["table_analysis"].get("tables", [])
        
        for current_table in current_tables:
            # Use previous chunk's column names
            if previous_tables:
                previous_table = previous_tables[0]  # Assuming first table for now
                current_table["column_names"] = previous_table.get("column_names", [])
        
        # Update title and description to match previous chunk
        chunk_data["title"] = previous_chunk.get("title", "")
    
    return chunk_data

def process_chunks_for_column_consistency(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Main function to process chunks and fix column names."""
    
    if not chunks:
        return chunks
    
    processed_chunks = []
    table_group_counter = 1
    current_table_group_id = None
    
    # Process first chunk (no previous chunk to compare)
    first_chunk = chunks[0].copy()
    current_table_group_id = f"table_group_{table_group_counter}"
    first_chunk["table_group_id"] = current_table_group_id
    processed_chunks.append(first_chunk)
    
    print(f"Processing chunk 1/{len(chunks)} - Assigned to {current_table_group_id}")
    
    # Process remaining chunks
    for i in range(1, len(chunks)):
        current_chunk = chunks[i].copy()
        previous_chunk = processed_chunks[-1]
        
        print(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Get previous column names
        previous_columns = extract_column_names_from_tables(previous_chunk)
        
        if not previous_columns:
            # If no previous columns, treat as new table
            table_group_counter += 1
            current_table_group_id = f"table_group_{table_group_counter}"
            current_chunk["table_group_id"] = current_table_group_id
            processed_chunks.append(current_chunk)
            continue
        
        # Get current chunk text
        previous_chunk_text = previous_chunk.get("chunk_text", "")
        current_chunk_text = current_chunk.get("chunk_text", "")

        llm_result = check_column_names_with_llm(previous_columns, previous_chunk_text, current_chunk_text)

        
        if llm_result["is_same_table"]:
            # Same table - update with previous chunk info
            print(f"  -> Same table detected (confidence: {llm_result['confidence']})")
            updated_chunk = update_chunk_with_same_table_info(
                current_chunk, previous_chunk, current_table_group_id
            )
            processed_chunks.append(updated_chunk)
        else:
            # Different table - create new group
            print(f"  -> Different table detected (confidence: {llm_result['confidence']})")
            table_group_counter += 1
            current_table_group_id = f"table_group_{table_group_counter}"
            current_chunk["table_group_id"] = current_table_group_id
            processed_chunks.append(current_chunk)
    
    return processed_chunks

def get_table_group_summary(processed_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary of table groups."""
    
    table_groups = {}
    
    for chunk in processed_chunks:
        group_id = chunk.get("table_group_id", "unknown")
        if group_id not in table_groups:
            table_groups[group_id] = {
                "chunk_count": 0,
                "title": chunk.get("title", ""),
                "column_names": extract_column_names_from_tables(chunk)
            }
        table_groups[group_id]["chunk_count"] += 1
    
    return {
        "total_table_groups": len(table_groups),
        "table_groups": table_groups
    }

def main(input_file: str, output_file: str):
    """Main function to process the JSON file."""
    
    try:
        # Load JSON data and extract chunks
        print(f"Loading data from {input_file}")
        metadata, chunks = load_json_data(input_file)
        
        print(f"Found {len(chunks)} chunks to process")
        
        # Process chunks for column consistency
        processed_chunks = process_chunks_for_column_consistency(chunks)
        
        # Save results with original structure
        save_json_data(metadata, processed_chunks, output_file)
        
        # Generate summary
        summary = get_table_group_summary(processed_chunks)
        
        print(f"\nProcessing complete! Results saved to {output_file}")
        print(f"Summary:")
        print(f"- Total table groups: {summary['total_table_groups']}")
        
        for group_id, info in summary['table_groups'].items():
            print(f"- {group_id}: {info['chunk_count']} chunks")
        
        return processed_chunks
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    input_file = "/Users/nilab/Desktop/projects/Knowladge-Base/processed_file.json"  # Your input JSON file
    output_file = "processed_chunks_fixed.json"  # Output file
    
    # Process the chunks
    result = main(input_file, output_file)