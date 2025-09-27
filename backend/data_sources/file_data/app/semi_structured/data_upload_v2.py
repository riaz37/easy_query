import os
import json
from pathlib import Path
from typing import Optional
from data_sources.file_data.app.semi_structured.chunker import process_file_with_chunking, llm, save_results_to_file, get_processing_summary
from data_sources.file_data.app.semi_structured.chunker_cleaner import process_chunks_for_column_consistency, load_json_data, save_json_data, get_table_group_summary
from data_sources.file_data.app.semi_structured.data_formater import process_json_to_chunks
from data_sources.file_data.app.semi_structured.parser import extract_markdown_from_file
import tempfile

def extract_filename_from_path(file_path: str) -> str:
    """Extract filename without extension from file path."""
    return Path(file_path).stem

def process_file_pipeline(
    file_path: str,
    preserve_layout_alignment_across_pages: bool = True,
    skip_diagonal_text: bool = True,
    output_tables_as_HTML: bool = False,
    disable_image_extraction: bool = False,
    spreadsheet_extract_sub_tables: bool = True,
    result_type: str = "markdown",
    chunk_size: int = 20000,
    chunk_overlap: int = 1000,
    user_id: Optional[str] = None,
    file_description: Optional[str] = None,
    table_name: Optional[str] = None
):
    """
    Process file through three main functions sequentially:
    1. Extract markdown from file (Excel, PDF, etc.)
    2. Process markdown with chunking
    3. Process chunks for column consistency  
    4. Process JSON to chunks
    """
    
    # Validate chunk parameters
    if chunk_overlap >= chunk_size:
        print(f"Error: chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        return None
    
    # Extract filename from path
    file_name = extract_filename_from_path(file_path)
    print(f"Processing file: {file_name}")
    print(f"Full path: {file_path}")
    print(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    
    # Step 0: Extract markdown from file
    print("\n=== STEP 0: Extracting markdown from file ===")
    
    try:
        markdown_text = extract_markdown_from_file(
            file_path,
            preserve_layout_alignment_across_pages=preserve_layout_alignment_across_pages,
            skip_diagonal_text=skip_diagonal_text,
            output_tables_as_HTML=output_tables_as_HTML,
            disable_image_extraction=disable_image_extraction,
            spreadsheet_extract_sub_tables=spreadsheet_extract_sub_tables,
            result_type=result_type
        )
        
        print(f"Successfully extracted markdown text ({len(markdown_text)} characters)")
        
    except Exception as e:
        print(f"Error in Step 0 (markdown extraction): {e}")
        return None
    
    # Step 1: Process markdown text with chunking (using temporary file)
    print("\n=== STEP 1: Processing markdown with chunking ===")
    
    try:
        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(markdown_text)
            temp_md_path = temp_file.name
        
        # Process the temporary markdown file with chunking
        # Pass the chunk parameters explicitly to ensure they're used
        print(f"Processing markdown content with chunking (size: {chunk_size}, overlap: {chunk_overlap})")
        processing_result = process_file_with_chunking(
            temp_md_path, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap, 
            llm=llm
        )
        
        # Clean up temporary file
        os.unlink(temp_md_path)
        
        # Generate summary
        summary = get_processing_summary(processing_result)
        print("Step 1 Processing Summary:")
        print(json.dumps(summary, indent=2))
        
        print(f"Step 1 completed successfully")
        
    except Exception as e:
        print(f"Error in Step 1: {e}")
        # Clean up temporary file if it exists
        if 'temp_md_path' in locals() and os.path.exists(temp_md_path):
            os.unlink(temp_md_path)
        return None
    
    # Step 2: Process chunks for column consistency
    print("\n=== STEP 2: Processing chunks for column consistency ===")
    
    try:
        # Extract chunks directly from processing_result
        chunks = processing_result.get('chunks', [])
        metadata = {k: v for k, v in processing_result.items() if k != 'chunks'}
        
        print(f"Found {len(chunks)} chunks to process")
        
        # Process chunks for column consistency
        processed_chunks = process_chunks_for_column_consistency(chunks)
        
        # Generate summary
        summary = get_table_group_summary(processed_chunks)
        print(f"Step 2 Processing complete!")
        print(f"Summary:")
        print(f"- Total table groups: {summary['total_table_groups']}")
        for group_id, info in summary['table_groups'].items():
            print(f"- {group_id}: {info['chunk_count']} chunks")
            
        print(f"Step 2 completed successfully")
        
    except Exception as e:
        print(f"Error in Step 2: {e}")
        return None
    
    # Step 3: Process JSON to chunks (using temporary file)
    print("\n=== STEP 3: Processing JSON to chunks ===")
    
    try:
        # Create temporary JSON file for the final step
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump({**metadata, 'chunks': processed_chunks}, temp_file, indent=2)
            temp_json_path = temp_file.name
        
        # Process JSON to chunks using the temporary file
        process_json_to_chunks(temp_json_path, file_name, user_id=user_id, file_description=file_description, table_name=table_name)
        
        # Clean up temporary file
        os.unlink(temp_json_path)
        
        print(f"Step 3 completed successfully")
        
    except Exception as e:
        print(f"Error in Step 3: {e}")
        # Clean up temporary file if it exists
        if 'temp_json_path' in locals() and os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return None
    
    print(f"\n🎉 Complete pipeline processing finished for: {file_name}")
    print(f"All processing completed without creating intermediate files")
    
    return True

def main():
    """Main function to run the complete pipeline."""
    
    # Example file path - replace with your actual file path
    file_path = "/Users/nilab/Desktop/projects/Knowladge-Base/app/semi-structured/RandomData.xlsx"
    
    # You can also get file path from command line arguments
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    # Validate file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Run the complete pipeline with your specified parameters
    # Fixed: Ensure chunk_overlap is smaller than chunk_size
    chunk_size = 20000
    chunk_overlap = 1000
    
    print(f"Starting pipeline with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    success = process_file_pipeline(
        file_path=file_path,
        preserve_layout_alignment_across_pages=False,  # Your setting
        skip_diagonal_text=True,
        output_tables_as_HTML=False,
        disable_image_extraction=False,
        spreadsheet_extract_sub_tables=True,
        result_type="markdown",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if success:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ Pipeline failed. Check error messages above.")

if __name__ == "__main__":
    main()